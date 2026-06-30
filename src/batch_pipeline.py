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

from src.single_pipeline import ensure_dirs, compute_dqi, analyse_single_target, _build_result_row

def analyse_batch(target_list=None, sector=None, max_targets=None,
                  classifier=None, sde_threshold=7.0):
    """
    Process multiple targets in batch mode.

    Provide either:
      - target_list: list of (tic_id, name) tuples
      - sector: TESS sector number (will download all targets)
    """
    ensure_dirs()
    all_results = []

    if sector is not None:
        logger.info(f"\n{'='*60}")
        logger.info(f"  BATCH PROCESSING: TESS Sector {sector}")
        logger.info(f"{'='*60}\n")

        for tic_id, time_arr, flux_arr, flux_err in download_sector_lightcurves(
            sector, max_targets=max_targets
        ):
            row = _process_single_lc(
                tic_id, f"TIC {tic_id}", time_arr, flux_arr, flux_err,
                classifier=classifier, sde_threshold=sde_threshold,
            )
            if row is not None:
                all_results.append(row)

    elif target_list is not None:
        logger.info(f"\n{'='*60}")
        logger.info(f"  BATCH PROCESSING: {len(target_list)} targets")
        logger.info(f"{'='*60}\n")

        for target_name in tqdm(target_list, desc="Processing"):
            try:
                rows = analyse_single_target(
                    target_name, max_sectors=3,
                    classifier=classifier,
                )
                if rows:
                    all_results.extend(rows)
            except Exception as e:
                logger.info(f"  ✗ {target_name}: {e}")
                continue
    else:
        raise ValueError("Provide either target_list or sector.")

    # Aggregate results
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Rank by DQI + Confidence + SDE
        if 'detection_significance' in results_df.columns and 'confidence_score' in results_df.columns and 'sde' in results_df.columns:
            results_df['rank_score'] = results_df['detection_significance'] + results_df['confidence_score'] + results_df['sde']
            results_df = results_df.sort_values('rank_score', ascending=False)
            
        csv_path = export_results_csv(results_df, RESULTS_DIR)

        if sector is not None:
            plot_sector_summary(results_df, sector, save_dir=PLOTS_DIR)

        generate_summary_report(results_df, RESULTS_DIR)

        logger.info(f"\n{'='*60}")
        logger.info(f"  BATCH COMPLETE: {len(all_results)} targets processed")
        logger.info(f"  Results saved to {RESULTS_DIR}/")
        logger.info(f"{'='*60}")
        return results_df

    logger.info("No results generated.")
    return pd.DataFrame()

def _process_single_lc(tic_id, target_name, time_arr, flux_arr, flux_err,
                       classifier=None, sde_threshold=7.0):
    """Process a single pre-downloaded light curve."""
    try:
        # Clean & detrend
        time_arr, flux_arr, flux_err = clean_lightcurve(
            time_arr, flux_arr, flux_err
        )
        if len(time_arr) < 100:
            return None

        # Two-pass transit-aware detrending (same logic as single pipeline)
        coarse_flat, _ = detrend_lightcurve(
            time_arr, flux_arr, method="biweight", window_length=DETREND_WINDOW
        )
        known = None
        try:
            first_det = run_tls(time_arr, coarse_flat)
            dur_hr = first_det.duration * 24
            if first_det.sde >= MIN_SDE and dur_hr <= 6.0:
                known = [(first_det.period, first_det.t0, first_det.duration)]
        except Exception:
            pass
        flat_flux, _ = detrend_transit_aware(
            time_arr, flux_arr, flux_err, window_length=DETREND_WINDOW,
            method="biweight", known_transits=known,
        )

        # Detect
        detection = run_tls(time_arr, flat_flux)

        if detection.sde < sde_threshold:
            return None  # Skip non-significant detections in batch

        # Feature extraction
        stellar = get_stellar_params(tic_id)
        features = extract_features(
            detection, time_arr, flat_flux,
            stellar_radius_rsun=stellar["radius_rsun"],
            stellar_teff=stellar["teff_k"],
            raw_flux=flux_arr,
        )

        # Classification
        if classifier is not None and classifier.is_trained:
            classification = classifier.predict(features)
        else:
            classification = classify_rule_based(features)

        # Fast fit only in batch mode
        fit_result = None
        try:
            fit_result = fit_transit_fast(
                time_arr, flat_flux, flux_err, detection=detection
            )
        except Exception:
            pass

        # Compute DQI
        det_significance = compute_dqi(detection, fit_result)

        # Generate diagnostic plot
        try:
            plot_candidate_diagnostic(
                time_arr, flat_flux, detection, fit_result,
                classification, features, target_name, save_dir=PLOTS_DIR,
                dqi=det_significance,
            )
        except Exception:
            pass

        return _build_result_row(
            tic_id, target_name, detection, classification, features, fit_result,
            detection_significance=det_significance
        )

    except Exception as e:
        return None

