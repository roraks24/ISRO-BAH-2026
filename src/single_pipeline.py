import logging
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import (
    download_light_curve, download_sector_lightcurves,
    load_curated_dataset, load_toi_catalog, get_stellar_params,
)
from src.config import MIN_SDE, DETREND_WINDOW
from src.detrending import (
    clean_lightcurve, detrend_lightcurve, detrend_transit_aware,
)
from src.detector import run_tls, run_multi_planet_search
from src.feature_extractor import extract_features
from src.classifier import TransitClassifier, classify_rule_based
from src.transit_fitter import fit_transit_fast, fit_transit_mcmc
from src.visualizer import (
    plot_candidate_diagnostic, plot_sector_summary,
    plot_corner, plot_individual_transits,
)
from src.report_generator import export_results_csv, generate_summary_report

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = "models"

logger = logging.getLogger(__name__)


def ensure_dirs():
    for d in [RESULTS_DIR, PLOTS_DIR, MODELS_DIR, "data"]:
        os.makedirs(d, exist_ok=True)

def compute_dqi(detection, fit_result=None):
    sde_score = min(detection.sde / 7.0, 3.0)
    snr_score = min(detection.snr / 7.1, 3.0)
    dqi = (sde_score * 0.4 + snr_score * 0.4) * 100
    if fit_result:
        if fit_result.fit_quality == 'Questionable':
            dqi -= 20
        elif fit_result.fit_quality == 'Failed':
            dqi -= 50
    if getattr(detection, 'warnings', []):
        dqi -= len(detection.warnings) * 10
    return max(min(dqi, 100.0), 0.0)

def analyse_single_target(target_name, max_sectors=5, do_mcmc=False,
                          classifier=None):
    """
    Full pipeline for a single target.
    """
    ensure_dirs()
    logger.info(f"\n{'='*60}")
    logger.info(f"  ANALYSING: {target_name}")
    logger.info(f"{'='*60}\n")

    # 1. Download
    logger.info("[1/6] Downloading light curve...")
    lc, tic_id = download_light_curve(target_name, max_sectors=max_sectors)

    # 2. Clean & detrend
    logger.info("[2/6] Cleaning and detrending...")
    time_arr = np.asarray(lc.time.value, dtype=np.float64)
    flux_arr = np.asarray(lc.flux.value, dtype=np.float64)
    try:
        flux_err_arr = np.asarray(lc.flux_err.value, dtype=np.float64)
    except Exception:
        flux_err_arr = None

    time_arr, flux_arr, flux_err_arr = clean_lightcurve(
        time_arr, flux_arr, flux_err_arr
    )

    # Two-pass, transit-aware detrending:
    #   Pass 1: coarse detrend, run a quick TLS to find the dominant transit,
    #   Pass 2: re-detrend with that transit masked so the trend is not pulled
    #           down by the dip (recovers true transit depth).
    # We only mask if the detected feature looks like a genuine transit.
    # Criteria for masking:
    #   - SDE >= threshold
    #   - Duration between 0.5 hr and 6 hr (real transits are never shorter than
    #     ~30 min or longer than ~6 hr for planets; longer features are stellar
    #     variability)
    #   - Depth < 1% (10000 ppm) — deeper signals are likely EBs whose shape
    #     must be preserved for downstream classification
    #   - Duration is NOT spuriously short relative to period.  TLS sometimes fits
    #     a narrow "transit" inside a broad sinusoidal variability bump.  If the
    #     reported duration would imply fewer than 2 complete transits, skip masking.
    coarse_flat, _ = detrend_lightcurve(time_arr, flux_arr, method="biweight",
                                         window_length=DETREND_WINDOW)
    known = None
    try:
        first_det = run_tls(time_arr, coarse_flat)
        dur_hr = first_det.duration * 24
        dur_is_realistic = dur_hr >= 0.5 and dur_hr <= 6.0
        depth_is_shallow = first_det.depth_ppm < 10000
        # Check the duration isn't spuriously short (TLS artifact on sinusoidal
        # variability).  For a 16-day signal, a real 0.64hr "transit" would mean
        # <1% of the orbit is in transit — physically implausible for a planet.
        dur_phase = first_det.duration / first_det.period if first_det.period > 0 else 0
        dur_not_spurious = dur_phase > 0.005  # > 0.5% of orbit
        if (first_det.sde >= MIN_SDE and dur_is_realistic
                and depth_is_shallow and dur_not_spurious):
            known = [(first_det.period, first_det.t0, first_det.duration)]
            logger.info(f"  Coarse detection: P={first_det.period:.3f}d "
                        f"depth={first_det.depth_ppm:.0f}ppm dur={dur_hr:.2f}hr "
                        f"— masking for transit-aware detrend.")
        elif first_det.sde >= MIN_SDE:
            reasons = []
            if not dur_is_realistic:
                reasons.append(f"dur={dur_hr:.2f}hr outside 0.5-6hr transit range")
            if not depth_is_shallow:
                reasons.append(f"depth={first_det.depth_ppm:.0f}ppm > 10000ppm (EB range)")
            if not dur_not_spurious:
                reasons.append(f"dur/period={dur_phase:.4f} suspiciously short")
            logger.info(f"  Coarse feature P={first_det.period:.3f}d — not masking "
                        f"({', '.join(reasons)}).")
    except Exception:
        known = None
    flat_flux, trend = detrend_transit_aware(
        time_arr, flux_arr, flux_err_arr, window_length=DETREND_WINDOW,
        method="biweight", known_transits=known,
    )
    logger.info(f"  Detrended {len(time_arr)} points "
                f"(transit-aware, window={DETREND_WINDOW}d).")

    # 3. Detect (iterative multi-planet search)
    logger.info("[3/6] Running multi-planet TLS detection...")
    detections = run_multi_planet_search(time_arr, flat_flux, max_planets=3)

    if not detections:
        logger.info(f"\n  ⚠ No significant signals found for {target_name}.")
        return []

    all_result_rows = []

    for planet_idx, detection in enumerate(detections):
        planet_label = f"planet {planet_idx + 1}" if len(detections) > 1 else ""
        if planet_label:
            logger.info(f"\n  ── {planet_label} ──")
        logger.info(f"  Period:   {detection.period:.4f} d")
        logger.info(f"  Depth:    {detection.depth_ppm:.0f} ppm")
        logger.info(f"  Duration: {detection.duration*24:.2f} hr")
        logger.info(f"  SDE:      {detection.sde:.1f}")
        logger.info(f"  SNR:      {detection.snr:.1f}")

        # DQI will be computed after model fitting

        if not detection.is_significant:
            logger.info(f"\n  ⚠ SDE={detection.sde:.1f} < 7 — below detection threshold.")
            logger.info("  Proceeding with analysis anyway for completeness...\n")

        # 4. Feature extraction & classification
        logger.info("[4/6] Extracting features and classifying...")
        stellar = get_stellar_params(tic_id)
        features = extract_features(
            detection, time_arr, flat_flux,
            stellar_radius_rsun=stellar["radius_rsun"],
            stellar_teff=stellar["teff_k"],
            raw_flux=flux_arr,
        )

        if classifier is not None and classifier.is_trained:
            classification = classifier.predict(features)
        else:
            classification = classify_rule_based(features)

        logger.info(f"  Verdict:    {classification.verdict}")
        logger.info(f"  Confidence: {classification.confidence_level} "
              f"({classification.confidence_score:.0f}/100)")
        for r in classification.reasons:
            logger.info(f"    → {r}")

        # 5. Transit model fitting
        logger.info("[5/6] Fitting transit model...")
        fit_result = None
        if detection.is_significant:
            try:
                if do_mcmc:
                    fit_result = fit_transit_mcmc(
                        time_arr, flat_flux, flux_err_arr,
                        detection=detection, nwalkers=32, nsteps=3000,
                    )
                else:
                    fit_result = fit_transit_fast(
                        time_arr, flat_flux, flux_err_arr,
                        detection=detection,
                    )
                logger.info(f"  Fitted Rp/Rs: {fit_result.rp_rs:.5f}")
                logger.info(f"  Fitted depth: {fit_result.depth_ppm:.0f} ppm")
                logger.info(f"  Fitted a/Rs:  {fit_result.a_rs:.2f}")
                logger.info(f"  Fitted inc:   {fit_result.inclination:.1f}°")
                logger.info(f"  χ²_red:       {fit_result.chi2_red:.3f}")
                if fit_result.rp_rs_err > 0:
                    logger.info(f"  Rp/Rs err:    ± {fit_result.rp_rs_err:.5f}")
            except Exception as e:
                logger.info(f"  ⚠ Fitting failed: {e}")
                fit_result = None

        # 5.5 Compute DQI
        det_significance = compute_dqi(detection, fit_result)
        logger.info(f"  Detection Quality Index (DQI): {det_significance:.1f}/100")

        # 6. Visualization
        logger.info("[6/6] Generating plots...")
        suffix = f"_p{planet_idx + 1}" if len(detections) > 1 else ""
        plot_target_name = (f"{target_name} ({planet_label})"
                           if planet_label else target_name)
        plot_candidate_diagnostic(
            time_arr, flat_flux, detection, fit_result,
            classification, features, plot_target_name, save_dir=PLOTS_DIR,
            dqi=det_significance,
        )

        plot_individual_transits(
            time_arr, flat_flux, detection, plot_target_name,
            save_dir=PLOTS_DIR,
        )

        if fit_result is not None and fit_result.mcmc_samples is not None:
            plot_corner(
                fit_result.mcmc_samples, fit_result.mcmc_labels,
                plot_target_name, save_dir=PLOTS_DIR,
            )

        # Build result row (includes detection significance)
        result_row = _build_result_row(
            tic_id, target_name, detection, classification, features,
            fit_result, planet_number=planet_idx + 1,
            detection_significance=det_significance,
        )

        # Print final summary
        _print_final_summary(plot_target_name, detection, fit_result,
                             classification)

        all_result_rows.append(result_row)

    return all_result_rows

def _build_result_row(tic_id, target_name, detection, classification,
                      features, fit_result, planet_number=1,
                      detection_significance=None):
    """Build a flat dict for the results DataFrame."""
    row = {
        "tic_id": tic_id,
        "target_name": target_name,
        "planet_number": planet_number,
        "period_days": detection.period,
        "duration_hr": detection.duration * 24,
        "depth_ppm": detection.depth_ppm,
        "sde": detection.sde,
        "snr": detection.snr,
        "n_transits": detection.n_transits,
        "odd_even_mismatch": detection.odd_even_mismatch,
        "detection_significance": detection_significance,
        "classification": classification.verdict,
        "confidence_level": classification.confidence_level,
        "confidence_score": classification.confidence_score,
        "classification_method": classification.method,
    }

    # Add fitted parameters if available
    if fit_result is not None:
        row.update({
            "fit_rp_rs": fit_result.rp_rs,
            "fit_rp_rs_err": fit_result.rp_rs_err,
            "fit_a_rs": fit_result.a_rs,
            "fit_quality": fit_result.fit_quality,
            "fit_inclination": fit_result.inclination,
            "fit_depth_ppm": fit_result.depth_ppm,
            "fit_duration_hr": fit_result.duration_hr,
            "fit_impact_parameter": fit_result.impact_parameter,
            "fit_chi2_red": fit_result.chi2_red,
            "fit_method": fit_result.method,
        })

    # Add key features
    for key in ["rp_earth_radii", "v_shape_metric", "secondary_eclipse_sigma",
                "rotation_near_transit_period"]:
        row[key] = features.get(key, np.nan)

    return row

def _print_final_summary(target_name, detection, fit_result, classification):
    """Print a clean final summary."""
    logger.info(f"\n{'═'*60}")
    logger.info(f"  RESULTS: {target_name}")
    logger.info(f"{'═'*60}")
    logger.info(f"")
    logger.info(f"  Classification:  {classification.verdict}")
    logger.info(f"  Confidence:      {classification.confidence_level} "
          f"({classification.confidence_score:.0f}/100)")
    logger.info(f"")
    logger.info(f"  ┌─ Detection ────────────────────────────────┐")
    logger.info(f"  │  Period:     {detection.period:.6f} days             │")
    logger.info(f"  │  Depth:      {detection.depth_ppm:>8.0f} ppm               │")
    logger.info(f"  │  Duration:   {detection.duration*24:>8.2f} hours             │")
    logger.info(f"  │  SDE:        {detection.sde:>8.1f}                     │")
    logger.info(f"  │  SNR:        {detection.snr:>8.1f}                     │")
    logger.info(f"  │  N_transits: {detection.n_transits:>8d}                     │")
    logger.info(f"  └────────────────────────────────────────────┘")

    if fit_result is not None:
        logger.info(f"")
        logger.info(f"  ┌─ Model Fit ({fit_result.method}) ─────────────────────┐")
        logger.info(f"  │  Rp/Rs:      {fit_result.rp_rs:>10.5f}"
              f"{'  ± ' + f'{fit_result.rp_rs_err:.5f}' if fit_result.rp_rs_err > 0 else '':>16}│")
        logger.info(f"  │  Depth:      {fit_result.depth_ppm:>10.0f} ppm             │")
        logger.info(f"  │  a/Rs:       {fit_result.a_rs:>10.2f}                  │")
        logger.info(f"  │  Inclination:{fit_result.inclination:>10.1f}°                 │")
        logger.info(f"  │  Impact b:   {fit_result.impact_parameter:>10.3f}                  │")
        logger.info(f"  │  χ²_red:     {fit_result.chi2_red:>10.3f}                  │")
        logger.info(f"  └────────────────────────────────────────────┘")

    logger.info(f"")
    logger.info(f"  Reasons:")
    for r in classification.reasons:
        logger.info(f"    • {r}")
    logger.info(f"{'═'*60}\n")

