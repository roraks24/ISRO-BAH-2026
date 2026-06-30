"""
synthetic_test.py — Offline demo of the full pipeline using synthetic light curves.

Generates three synthetic TESS-like light curves:
  1. A genuine planet transit (Pi Men c parameters)
  2. An eclipsing binary (deep, V-shaped, odd-even mismatch)
  3. A stellar rotation / false alarm (sinusoidal modulation)

Runs each through the full detection → feature extraction → classification →
transit fitting → visualisation pipeline WITHOUT any MAST download.

Usage:
    python synthetic_test.py
"""

import sys
import os
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings("ignore")

from src.detector import run_tls, DetectionResult
from src.detrending import clean_lightcurve, detrend_lightcurve, detrend_transit_aware
from src.feature_extractor import extract_features
from src.classifier import classify_rule_based
from src.transit_fitter import fit_transit_fast
from src.visualizer import plot_candidate_diagnostic, plot_individual_transits
from src.report_generator import export_results_csv, generate_summary_report

try:
    import batman
    HAS_BATMAN = True
except ImportError:
    HAS_BATMAN = False


# ── Constants ──────────────────────────────────────────────────────────

RNG = np.random.default_rng(42)
CADENCE_MIN = 2        # TESS 2-min cadence
DURATION_DAYS = 27     # single TESS sector
N_POINTS = int(DURATION_DAYS * 24 * 60 / CADENCE_MIN)
TIME_BASE = np.linspace(0, DURATION_DAYS, N_POINTS)
NOISE_LEVEL = 250e-6   # 250 ppm shot noise (typical TESS Tmag~10)

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ── Synthetic light curve generators ──────────────────────────────────

def _batman_transit(time, period, t0, rp, a, inc, u1=0.4, u2=0.2):
    """Generate a batman transit model."""
    if not HAS_BATMAN:
        return np.ones_like(time)
    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp
    params.a = a
    params.inc = inc
    params.ecc = 0.0
    params.w = 90.0
    params.u = [u1, u2]
    params.limb_dark = "quadratic"
    m = batman.TransitModel(params, time)
    return m.light_curve(params)


def make_planet_lc(period=6.268, rp_rs=0.0163, a_rs=16.8, inc=89.5,
                   noise=NOISE_LEVEL):
    """Genuine planet transit: Pi Men c-like parameters."""
    t0 = period * 0.3
    flux = _batman_transit(TIME_BASE, period, t0, rp_rs, a_rs, inc)
    flux += RNG.normal(0, noise, size=len(TIME_BASE))
    flux_err = np.full_like(flux, noise)
    return TIME_BASE.copy(), flux, flux_err


def make_eb_lc(period=3.14, depth_primary=0.08, depth_secondary=0.06,
               noise=NOISE_LEVEL):
    """
    Eclipsing binary: two eclipses per period (primary + secondary),
    V-shaped profile, significant odd-even depth mismatch.
    """
    flux = np.ones(N_POINTS)
    phase = ((TIME_BASE - 0.1) % period) / period

    # Primary eclipse (deep, V-shaped)
    in_primary = np.abs(phase - 0.0) < 0.015
    in_primary |= np.abs(phase - 1.0) < 0.015
    dist_primary = np.minimum(np.abs(phase), np.abs(phase - 1.0))
    flux -= depth_primary * np.clip(1.0 - dist_primary / 0.015, 0, 1)

    # Secondary eclipse at phase 0.5 (slightly shallower → EB signature)
    dist_secondary = np.abs(phase - 0.5)
    in_secondary = dist_secondary < 0.012
    flux -= depth_secondary * np.clip(1.0 - dist_secondary / 0.012, 0, 1)

    flux += RNG.normal(0, noise, size=N_POINTS)
    flux_err = np.full_like(flux, noise)
    return TIME_BASE.copy(), flux, flux_err


def make_starspot_lc(rotation_period=12.0, amplitude=0.003, noise=NOISE_LEVEL):
    """Stellar rotation / starspot: quasi-sinusoidal with no sharp dip."""
    flux = (1.0
            + amplitude * np.sin(2 * np.pi * TIME_BASE / rotation_period)
            + 0.3 * amplitude * np.sin(4 * np.pi * TIME_BASE / rotation_period + 0.7))
    flux += RNG.normal(0, noise, size=N_POINTS)
    flux_err = np.full_like(flux, noise)
    return TIME_BASE.copy(), flux, flux_err


# ── Run pipeline on one synthetic LC ──────────────────────────────────

def run_on_synthetic(name, time, flux, flux_err,
                     stellar_radius=1.0, teff=5778.0):
    print(f"\n{'='*60}")
    print(f"  SYNTHETIC TEST: {name}")
    print(f"{'='*60}")

    # 0. Clean + transit-aware detrend (matches the real pipeline)
    print("[0/4] Cleaning and transit-aware detrending...")
    time, flux, flux_err = clean_lightcurve(time, flux, flux_err)
    coarse_flat, _ = detrend_lightcurve(time, flux, method="biweight",
                                         window_length=1.5)
    first_det = run_tls(time, coarse_flat)
    known = None
    dur_hr = first_det.duration * 24
    if first_det.sde >= 7.0 and dur_hr <= 6.0:
        known = [(first_det.period, first_det.t0, first_det.duration)]
    raw_flux = flux.copy()
    flux, _ = detrend_transit_aware(time, flux, flux_err,
                                     window_length=1.5, known_transits=known)

    # 1. Detect
    print("[1/4] Running TLS detection...")
    detection = run_tls(time, flux)
    print(f"  Period:   {detection.period:.4f} d")
    print(f"  Depth:    {detection.depth_ppm:.0f} ppm")
    print(f"  Duration: {detection.duration*24:.2f} hr")
    print(f"  SDE:      {detection.sde:.1f}  |  SNR: {detection.snr:.1f}")

    # 2. Features + classify
    print("[2/4] Extracting features and classifying...")
    features = extract_features(
        detection, time, flux,
        stellar_radius_rsun=stellar_radius, stellar_teff=teff,
        raw_flux=raw_flux,
    )
    classification = classify_rule_based(features)
    print(f"  Verdict:    {classification.verdict}")
    print(f"  Confidence: {classification.confidence_level} ({classification.confidence_score:.0f}/100)")
    for r in classification.reasons:
        print(f"    -> {r}")

    # 3. Fit transit model
    print("[3/4] Fitting transit model...")
    fit_result = None
    if detection.sde > 5.0:
        try:
            fit_result = fit_transit_fast(time, flux, flux_err, detection=detection)
            print(f"  Rp/Rs:  {fit_result.rp_rs:.5f}")
            print(f"  a/Rs:   {fit_result.a_rs:.2f}")
            print(f"  inc:    {fit_result.inclination:.1f} deg")
            print(f"  chi2_red: {fit_result.chi2_red:.3f}")
        except Exception as e:
            print(f"  Fit failed: {e}")

    # 4. Plots
    print("[4/4] Generating plots...")
    plot_candidate_diagnostic(
        time, flux, detection, fit_result,
        classification, features, name, save_dir="results/plots"
    )
    plot_individual_transits(
        time, flux, detection, name, save_dir="results/plots"
    )

    return {
        "target_name": name,
        "period_days": detection.period,
        "depth_ppm": detection.depth_ppm,
        "duration_hr": detection.duration * 24,
        "sde": detection.sde,
        "snr": detection.snr,
        "n_transits": detection.n_transits,
        "odd_even_mismatch": detection.odd_even_mismatch,
        "classification": classification.verdict,
        "confidence_level": classification.confidence_level,
        "confidence_score": classification.confidence_score,
        "classification_method": classification.method,
        "fit_rp_rs": fit_result.rp_rs if fit_result else None,
        "fit_chi2_red": fit_result.chi2_red if fit_result else None,
        "rp_earth_radii": features.get("rp_earth_radii", None),
        "v_shape_metric": features.get("v_shape_metric", None),
        "secondary_eclipse_sigma": features.get("secondary_eclipse_sigma", None),
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("\nGenerating synthetic TESS-like light curves...")
    print(f"  Cadence:  {CADENCE_MIN} min  |  Duration: {DURATION_DAYS} days  |  N={N_POINTS:,} points")
    print(f"  Noise:    {NOISE_LEVEL*1e6:.0f} ppm  |  batman: {'yes' if HAS_BATMAN else 'no (box model)'}")

    cases = [
        ("Planet_Pi-Men-c",
         *make_planet_lc(),
         {"stellar_radius": 1.1, "teff": 6040}),
        ("EclipsingBinary_Synthetic",
         *make_eb_lc(),
         {"stellar_radius": 1.2, "teff": 5500}),
        ("Starspot_Synthetic",
         *make_starspot_lc(),
         {"stellar_radius": 0.9, "teff": 4800}),
    ]

    results = []
    for name, time, flux, flux_err, kwargs in cases:
        row = run_on_synthetic(name, time, flux, flux_err, **kwargs)
        results.append(row)

    # Summary
    print(f"\n{'='*60}")
    print("  SYNTHETIC TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Target':<28} {'SDE':>6} {'SNR':>6} {'Classification':<22} {'Conf':>5}")
    print(f"  {'-'*70}")
    for r in results:
        print(f"  {r['target_name']:<28} {r['sde']:>6.1f} {r['snr']:>6.1f} "
              f"{r['classification']:<22} {r['confidence_score']:>5.0f}")
    print(f"{'='*60}")

    df = pd.DataFrame(results)
    df.to_csv("results/synthetic_test_results.csv", index=False)
    print(f"\nResults saved to results/synthetic_test_results.csv")
    print("Plots saved to results/plots/")


if __name__ == "__main__":
    main()
