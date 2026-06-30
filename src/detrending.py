"""
detrending.py — Light curve detrending and cleaning.

Provides:
  - detrend_transit_aware()  — headline two-pass detrender that uses wotan's
    mask-aware biweight filter to preserve transit depth integrity.
  - detrend_lightcurve()     — one-pass convenience wrapper.
  - clean_lightcurve()       — NaN removal + sigma clipping.

Key principle: the trend must never be contaminated by transit dips, or the
transit depth is systematically under-recovered (a ~30% error in the old code).
wotan's flatten(mask=...) handles this natively and runs in C.
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter as _scipy_median_filter

try:
    from wotan import flatten as _wotan_flatten
    HAS_WOTAN = True
except Exception:
    HAS_WOTAN = False


def detrend_transit_aware(time, flux, flux_err=None, window_length=1.5,
                          method="biweight", known_transits=None):
    """
    Transit-preserving detrending using wotan's mask-aware biweight filter.

    1. Build a boolean mask that is True at in-transit points.
    2. Pass mask to wotan flatten(), which excludes those points when
       computing the trend.
    3. Divide the original flux by the mask-cleaned trend.

    When known_transits is None, an automatic dip detector is used.

    Parameters
    ----------
    time : array
        Time stamps (days).
    flux : array
        Flux values (normalised to ~1).
    flux_err : array or None
        Passed through unchanged.
    window_length : float
        Biweight window in days.  Default 1.5d (longer than typical TESS
        transits of 1–4 hours, so the trend is not pulled down).
    known_transits : list of (period, t0, duration_days) or None
        Transits to mask.  If None, dips are auto-detected.
    method : str
        Currently only 'biweight' is supported for the transit-aware path.

    Returns
    -------
    flat_flux, trend : arrays
    """
    time = np.asarray(time, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    if len(time) < 50:
        return flux.copy(), np.ones_like(flux)

    # Build mask: True = points to EXCLUDE from the trend fit
    transit_mask = _build_transit_mask(time, flux, known_transits)

    if HAS_WOTAN:
        flat, trend = _wotan_flatten(
            time, flux,
            window_length=window_length,
            method=method,
            return_trend=True,
            mask=transit_mask,
        )
    else:
        # Fallback: interpolate masked points, then run fast median + smooth
        trend = _detrend_masked_fallback(time, flux, transit_mask,
                                          window_days=window_length)

    trend = np.asarray(trend, dtype=np.float64)
    trend[trend == 0] = 1.0
    flat_flux = flux / trend
    return flat_flux, trend


def detrend_lightcurve(time, flux, flux_err=None, method="biweight",
                       window_length=0.5):
    """
    One-pass detrending (legacy, no transit protection).
    Used internally for the coarse first pass.
    """
    if method == "biweight" and HAS_WOTAN:
        _flat, trend = _wotan_flatten(time, flux, window_length=window_length,
                                       method="biweight", return_trend=True)
    elif method == "biweight":
        trend = _biweight_filter(time, flux, window_days=window_length)
    elif method == "savgol":
        wl = int(window_length) if window_length > 1 else 901
        if wl % 2 == 0:
            wl += 1
        trend = savgol_filter(flux, window_length=wl, polyorder=3)
    elif method == "median":
        trend = _median_filter(time, flux, window_days=window_length)
    else:
        raise ValueError(f"Unknown detrending method: {method}")

    trend = np.asarray(trend, dtype=np.float64)
    trend[trend == 0] = 1.0
    flat_flux = flux / trend
    return flat_flux, trend


# ── Internal helpers ────────────────────────────────────────────────────────

def _build_transit_mask(time, flux, known_transits):
    """
    Build a boolean mask (True = in-transit, should be excluded from trend).

    Combines explicit known_transits with automatic dip detection on the
    flux array itself. Deep eclipsing-binary eclipses are deliberately NOT
    masked (they carry the classification signal).
    """
    n = len(time)
    mask = np.zeros(n, dtype=bool)
    med = np.median(flux)
    mad_global = np.median(np.abs(flux - med))

    # (a) Explicitly mask known transits (primary mechanism)
    if known_transits:
        for period, t0, dur in known_transits:
            if period <= 0:
                continue
            phase = ((time - t0) % period) / period
            phase[phase > 0.5] -= 1.0
            # Mask 1.5× the transit duration for safety
            half_dur_phase = (dur / period) * 1.5
            mask |= np.abs(phase) < half_dur_phase

    # (b) Auto-detect SHALLOW dips only (planet-like, < 1% depth).
    # We deliberately use a deep threshold (5 MAD) so that we don't mask
    # deep eclipsing-binary eclipses — those are part of the signal we want
    # to preserve for downstream classification. Only mask points that look
    # like shallow planet transits contaminating the trend estimate.
    if mad_global > 1e-12:
        threshold = med - 5.0 * 1.4826 * mad_global
        shallow_mask = (flux < threshold) & (flux > med - 0.01)  # < 1% depth
        mask |= shallow_mask

    # Guard: never mask more than 15% of data (prevent catastrophic failures
    # where a variable star or EB gets almost entirely masked)
    if mask.sum() > 0.15 * n:
        # If we're masking too much, drop the auto-detected dips and keep
        # only the explicit known_transits mask
        mask = np.zeros(n, dtype=bool)
        if known_transits:
            for period, t0, dur in known_transits:
                if period <= 0:
                    continue
                phase = ((time - t0) % period) / period
                phase[phase > 0.5] -= 1.0
                half_dur_phase = (dur / period) * 1.5
                mask |= np.abs(phase) < half_dur_phase

    return mask


def _detrend_masked_fallback(time, flux, transit_mask, window_days=1.5):
    """
    Fast fallback detrending when wotan is unavailable.

    Interpolates in-transit points, then applies a median filter
    smoothed with a Savitzky-Golay pass to remove step artifacts.
    """
    n = len(time)
    flux_work = flux.copy()

    if transit_mask.any():
        oot = ~transit_mask
        flux_work[transit_mask] = np.interp(
            time[transit_mask], time[oot], flux[oot]
        )

    # Median filter (window in samples, not days)
    dt = np.median(np.diff(time))
    win_samples = max(int(window_days / dt), 15)
    if win_samples % 2 == 0:
        win_samples += 1
    trend = _scipy_median_filter(flux_work, size=win_samples, mode='reflect')
    # Smooth the median with a Savitzky-Golay pass
    if win_samples >= 7:
        trend = savgol_filter(trend, window_length=win_samples, polyorder=3)
    return trend


def _biweight_filter(time, flux, window_days=0.5):
    """
    Tukey's biweight (bisquare) location estimator applied in a sliding
    time window. Robust to outliers and transit dips — the standard for
    transit photometry detrending (cf. wotan).

    Uses wotan's optimised implementation when available.
    """
    if HAS_WOTAN:
        try:
            return np.asarray(
                _wotan_biweight(flux, window_length=window_days,
                                edge_cutoff=0.0),
                dtype=np.float64,
            )
        except Exception:
            pass

    n = len(time)
    trend = np.ones(n)
    half_win = window_days / 2.0

    for i in range(n):
        mask = np.abs(time - time[i]) <= half_win
        if mask.sum() < 5:
            trend[i] = np.median(flux[mask]) if mask.sum() > 0 else 1.0
            continue
        trend[i] = _biweight_location(flux[mask])

    return trend


def _biweight_location(data, c=6.0, max_iter=10):
    """
    Compute the biweight location estimator (robust mean).
    c = tuning constant (6.0 is standard for transit work).
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad < 1e-12:
        return median

    for _ in range(max_iter):
        u = (data - median) / (c * mad)
        mask = np.abs(u) < 1.0
        if mask.sum() == 0:
            return median
        w = (1 - u[mask] ** 2) ** 2
        new_median = np.sum(w * data[mask]) / np.sum(w)
        if abs(new_median - median) < 1e-10:
            break
        median = new_median

    return median


def _median_filter(time, flux, window_days=0.5):
    """Simple sliding median filter in time."""
    n = len(time)
    trend = np.ones(n)
    half_win = window_days / 2.0
    for i in range(n):
        mask = np.abs(time - time[i]) <= half_win
        if mask.sum() > 0:
            trend[i] = np.median(flux[mask])
    return trend


def sigma_clip_iterative(flux, sigma=3.0, max_iter=5):
    """
    Iterative sigma clipping — removes outliers beyond `sigma`
    standard deviations from the median, iterating until convergence.

    Returns a boolean mask (True = keep).
    """
    mask = np.ones(len(flux), dtype=bool)
    for _ in range(max_iter):
        med = np.median(flux[mask])
        std = np.std(flux[mask])
        if std < 1e-12:
            break
        new_mask = np.abs(flux - med) < sigma * std
        if np.array_equal(mask, new_mask):
            break
        mask = new_mask
    return mask


def mask_transits(time, period, t0, duration, width_factor=1.5):
    """
    Create a boolean mask that is True for out-of-transit points.
    Useful for detrending after an initial transit detection.

    width_factor : float
        Multiplier on the duration for safety margin.
    """
    half_dur = duration * width_factor / 2.0
    phase = ((time - t0) % period) / period
    # Wrap to [-0.5, 0.5]
    phase[phase > 0.5] -= 1.0
    in_transit = np.abs(phase * period) < half_dur
    return ~in_transit


def clean_lightcurve(time, flux, flux_err=None, sigma_clip=20.0):
    """
    One-shot cleaning: remove NaNs/Infs and clip only extreme outliers.

    IMPORTANT: the sigma threshold defaults to 20 (very conservative).
    A low threshold (e.g. 5) clips genuine transit/eclipse dips as if they
    were outliers, destroying the signal we are trying to detect. We only
    remove points that are far enough out to be cosmic rays or instrumental
    glitches, and we clip SYMMETRICALLY but with a high threshold so that
    real astrophysical dips (planets at ~0.01-1%, EBs up to ~50%) survive.

    Returns cleaned (time, flux, flux_err) arrays.
    """
    mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        mask &= np.isfinite(flux_err)

    time, flux = time[mask], flux[mask]
    if flux_err is not None:
        flux_err = flux_err[mask]

    # Normalise to median
    med = np.median(flux)
    if med > 0:
        flux = flux / med
        if flux_err is not None:
            flux_err = flux_err / med

    # Conservative sigma clip — only remove extreme outliers (cosmic rays,
    # detector glitches). Default sigma=20 preserves all real transits/EBs.
    clip_mask = sigma_clip_iterative(flux, sigma=sigma_clip)
    time, flux = time[clip_mask], flux[clip_mask]
    if flux_err is not None:
        flux_err = flux_err[clip_mask]

    return time, flux, flux_err
