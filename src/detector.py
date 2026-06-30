"""
detector.py — Transit signal detection using TLS and BLS.

Provides:
  - TLS (Transit Least Squares) detection with realistic transit shapes
  - BLS (Box Least Squares) as a faster complementary method
  - Multi-planet iterative search (detect, mask, re-detect)
  - Independent SNR computation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from transitleastsquares import transitleastsquares


@dataclass
class DetectionResult:
    """Standardised container for a single transit detection."""
    period: float = 0.0                 # days
    t0: float = 0.0                     # epoch (BJD)
    duration: float = 0.0               # days
    depth: float = 0.0                  # fractional (1 - min_flux)
    depth_ppm: float = 0.0              # parts per million
    rp_rs: float = 0.0                  # planet-to-star radius ratio
    sde: float = 0.0                    # Signal Detection Efficiency
    snr: float = 0.0                    # signal-to-noise ratio
    fap: float = 1.0                    # false alarm probability
    odd_even_mismatch: float = 0.0      # sigma
    n_transits: int = 0                 # number of observed transits
    transit_times: list = field(default_factory=list)
    # Raw TLS/BLS results for downstream use
    raw_results: object = None

    @property
    def is_significant(self):
        return self.sde >= 7.0


def run_tls(time, flux, period_min=0.5, period_max=20.0,
            show_progress=False, **kwargs):
    """
    Run Transit Least Squares on a cleaned, detrended light curve.

    Returns a DetectionResult.
    """
    model = transitleastsquares(time, flux)
    results = model.power(
        period_min=period_min,
        period_max=period_max,
        show_progress_bar=show_progress,
        **kwargs,
    )

    transit_times = list(np.atleast_1d(
        getattr(results, "transit_times", [])
    ))

    distinct_transits = int(getattr(results, "distinct_transit_count", len(transit_times)))

    det = DetectionResult(
        period=float(results.period),
        t0=float(results.T0),
        duration=float(results.duration),
        depth=float(1 - results.depth),
        depth_ppm=float((1 - results.depth) * 1e6),
        rp_rs=float(getattr(results, "rp_rs", np.sqrt(1 - results.depth))),
        sde=float(results.SDE),
        snr=float(results.snr),
        fap=float(getattr(results, "FAP", 1.0)),
        odd_even_mismatch=float(results.odd_even_mismatch) if np.isfinite(results.odd_even_mismatch) else 0.0,
        n_transits=distinct_transits,
        transit_times=transit_times,
        raw_results=results,
    )
    return det


def run_bls(time, flux, period_min=0.5, period_max=20.0,
            duration_min=0.01, duration_max=0.3):
    """
    Run Box Least Squares as a complementary/faster detection method.

    Returns a DetectionResult.
    """
    from astropy.timeseries import BoxLeastSquares

    bls = BoxLeastSquares(time, flux)
    durations = np.linspace(duration_min, duration_max, 20)
    results = bls.autopower(durations, minimum_period=period_min,
                            maximum_period=period_max)

    best_idx = np.argmax(results.power)
    best_period = float(results.period[best_idx])
    best_power = float(results.power[best_idx])

    # Compute SDE-like metric
    mean_power = np.mean(results.power)
    std_power = np.std(results.power)
    sde = (best_power - mean_power) / std_power if std_power > 0 else 0.0

    # Get transit parameters at best period
    stats = bls.compute_stats(best_period, results.duration[best_idx],
                              results.transit_time[best_idx])

    depth = float(stats.get("depth", [0])[0]) if "depth" in stats else 0.0

    # Compute SNR
    snr = _compute_snr(time, flux, best_period,
                       float(results.transit_time[best_idx]),
                       float(results.duration[best_idx]))

    det = DetectionResult(
        period=best_period,
        t0=float(results.transit_time[best_idx]),
        duration=float(results.duration[best_idx]),
        depth=depth,
        depth_ppm=depth * 1e6,
        rp_rs=np.sqrt(depth) if depth > 0 else 0.0,
        sde=sde,
        snr=snr,
        fap=1.0,  # BLS doesn't provide FAP directly
        odd_even_mismatch=0.0,
        n_transits=0,
        transit_times=[],
        raw_results=results,
    )
    return det


def run_multi_planet_search(time, flux, max_planets=3,
                            sde_threshold=7.0, **tls_kwargs):
    """
    Iterative multi-planet search: detect the strongest signal,
    mask it from the light curve, then search again.

    Returns a list of DetectionResults.
    """
    detections: List[DetectionResult] = []
    residual_flux = flux.copy()

    for i in range(max_planets):
        det = run_tls(time, residual_flux, **tls_kwargs)

        if det.sde < sde_threshold:
            break

        # Stop if period duplicates an already-found planet (±2%)
        is_duplicate = any(
            abs(det.period - prev.period) / prev.period < 0.02
            for prev in detections
        )
        if is_duplicate:
            break

        detections.append(det)

        # Mask the detected signal before searching for the next planet
        residual_flux = _mask_and_interpolate(
            time, residual_flux, det.period, det.t0, det.duration
        )

    return detections


def _mask_and_interpolate(time, flux, period, t0, duration):
    """
    Replace in-transit data points with locally interpolated baseline values
    using nearest out-of-transit points so subsequent TLS runs don't re-detect
    the same signal.
    """
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    half_dur_phase = (duration / period) / 2.0 * 1.5  # 1.5x safety margin

    in_transit = np.abs(phase) < half_dur_phase
    
    if not np.any(in_transit):
        return flux

    flux_new = flux.copy()
    out_of_transit = ~in_transit

    if out_of_transit.sum() > 0:
        flux_new[in_transit] = np.interp(
            time[in_transit],
            time[out_of_transit],
            flux[out_of_transit]
        )
        return flux_new
    return flux


def _compute_snr(time, flux, period, t0, duration):
    """
    Compute transit SNR independently:
        SNR = (depth) / (scatter / sqrt(n_in_transit))
    """
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    half_dur_phase = (duration / period) / 2.0

    in_transit = np.abs(phase) < half_dur_phase
    out_of_transit = ~in_transit

    if in_transit.sum() < 3 or out_of_transit.sum() < 10:
        return 0.0

    baseline = np.median(flux[out_of_transit])
    depth = baseline - np.median(flux[in_transit])
    scatter = np.std(flux[out_of_transit])

    if scatter < 1e-12:
        return 0.0

    snr = depth / (scatter / np.sqrt(in_transit.sum()))
    return float(snr)


def compute_detection_significance(det: DetectionResult):
    """
    Compute a combined significance metric that considers SDE, SNR,
    and number of transits.
    """
    sde_score = min(det.sde / 7.0, 3.0)  # normalise to threshold
    snr_score = min(det.snr / 7.1, 3.0)
    transit_bonus = min(det.n_transits / 3.0, 2.0)  # more transits = more confident

    significance = (sde_score * 0.4 + snr_score * 0.4 + transit_bonus * 0.2) * 100
    return min(significance, 100.0)
