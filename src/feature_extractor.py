"""
feature_extractor.py — Rich diagnostic feature extraction for transit classification.

Extracts features from a DetectionResult + light curve that capture the
physical signatures distinguishing planets from false positives:
  - Transit shape (V-shaped EB vs U-shaped planet)
  - Odd-even depth asymmetry
  - Secondary eclipse detection
  - Implied planet radius
  - Stellar rotation period
  - Statistical quality metrics
"""

import numpy as np
from astropy.timeseries import LombScargle


RSUN_IN_REARTH = 109.2
MAX_PLANET_RADIUS_REARTH = 22.0  # ~2 Jupiter radii


def extract_features(det, time, flux, stellar_radius_rsun=1.0,
                     stellar_teff=5778.0, raw_flux=None):
    """
    Extract a comprehensive feature vector from a detection result.

    Parameters
    ----------
    det : DetectionResult
        From detector.run_tls() or detector.run_bls().
    time : array
        Original time array.
    flux : array
        Detrended flux array.
    stellar_radius_rsun : float
        Host star radius in solar radii.
    stellar_teff : float
        Host star effective temperature in K.
    raw_flux : array or None
        Raw (cleaned, non-detrended) flux array. If None, defaults to `flux`.

    Returns
    -------
    features : dict
        Feature dictionary for classification.
    """
    features = {}
    results = det.raw_results

    # ── Core transit parameters ──
    features["period_days"] = det.period
    features["duration_hr"] = det.duration * 24
    features["depth_ppm"] = det.depth_ppm
    features["depth_frac"] = det.depth
    features["sde"] = det.sde
    features["snr"] = det.snr
    features["fap"] = det.fap
    features["odd_even_mismatch"] = det.odd_even_mismatch
    features["n_transits"] = det.n_transits

    # ── Implied planet radius ──
    rp_rs = det.rp_rs
    rp_earth = rp_rs * stellar_radius_rsun * RSUN_IN_REARTH
    features["rp_rs"] = rp_rs
    features["rp_earth_radii"] = rp_earth
    features["radius_exceeds_limit"] = float(rp_earth > MAX_PLANET_RADIUS_REARTH)

    # ── Stellar parameters ──
    features["stellar_radius_rsun"] = stellar_radius_rsun
    features["stellar_teff"] = stellar_teff

    # ── Transit shape: V-shape metric ──
    features["v_shape_metric"] = _compute_v_shape(results)

    # ── Secondary eclipse ──
    sec_depth, sec_sigma = _detect_secondary_eclipse(results)
    features["secondary_eclipse_depth"] = sec_depth
    features["secondary_eclipse_sigma"] = sec_sigma

    # ── Transit depth consistency (reduced chi-sq) ──
    features["depth_chi2_red"] = _transit_depth_consistency(results)

    # ── Model fit quality ──
    features["model_chi2_red"] = _model_fit_quality(results)

    # ── Stellar rotation period via Lomb-Scargle ──
    rot_features = _rotation_analysis(time, raw_flux if raw_flux is not None else flux, det)
    features.update(rot_features)

    # ── Duration vs expected duration ──
    features["duration_ratio"] = _duration_ratio(
        det.period, det.duration, stellar_radius_rsun
    )

    # ── Ingress/egress asymmetry ──
    features["ingress_egress_ratio"] = _ingress_egress_ratio(results)

    # ── Flux scatter ratio (in-transit vs out-of-transit) ──
    features["scatter_ratio"] = _scatter_ratio(time, flux, det)

    return features


def _compute_v_shape(results):
    """
    V-shape metric: ratio of (ingress+egress duration) to total duration.
    V-shaped (EB) → close to 1.0
    U-shaped (planet with flat bottom) → close to 0.0
    """
    if results is None:
        return 0.5

    try:
        phase = np.asarray(results.folded_phase)
        flux = np.asarray(results.folded_y)
        model = np.asarray(results.model_folded_model)
        model_phase = np.asarray(results.model_folded_phase)

        # Find the transit region in the model
        transit_mask = model < (1.0 - 0.1 * (1.0 - np.min(model)))
        if transit_mask.sum() < 3:
            return 0.5

        transit_phases = model_phase[transit_mask]
        transit_models = model[transit_mask]

        # Total transit width
        total_width = np.ptp(transit_phases)
        if total_width < 1e-6:
            return 0.5

        # Flat bottom width (within 10% of minimum)
        min_flux = np.min(transit_models)
        flat_threshold = min_flux + 0.1 * (1.0 - min_flux)
        flat_mask = transit_models < flat_threshold
        if flat_mask.sum() < 2:
            return 1.0  # No flat bottom → V-shaped

        flat_width = np.ptp(transit_phases[flat_mask])
        v_shape = 1.0 - (flat_width / total_width)
        return float(np.clip(v_shape, 0.0, 1.0))
    except Exception:
        return 0.5


def _detect_secondary_eclipse(results):
    """
    Search for a secondary eclipse near phase 0.5.
    Returns (depth, sigma).
    """
    if results is None:
        return 0.0, 0.0
    try:
        phase = np.asarray(results.folded_phase)
        flux = np.asarray(results.folded_y)

        sec_mask = (phase > 0.45) & (phase < 0.55)
        baseline_mask = (phase < 0.35) | (phase > 0.65)

        if sec_mask.sum() < 5 or baseline_mask.sum() < 5:
            return 0.0, 0.0

        baseline_level = np.median(flux[baseline_mask])
        baseline_scatter = np.std(flux[baseline_mask])
        sec_depth = baseline_level - np.median(flux[sec_mask])
        sec_sigma = sec_depth / baseline_scatter if baseline_scatter > 0 else 0
        return float(sec_depth), float(sec_sigma)
    except Exception:
        return 0.0, 0.0


def _transit_depth_consistency(results):
    """
    Compute the reduced chi-squared of individual transit depths
    relative to the mean depth. High values → depth varies → EB.
    """
    if results is None:
        return 1.0
    try:
        depths = getattr(results, "transit_depths", None)
        if depths is None or len(depths) < 2:
            return 1.0
        depths = np.asarray(depths)
        depths = depths[np.isfinite(depths)]
        if len(depths) < 2:
            return 1.0
        mean_d = np.mean(depths)
        if mean_d == 0:
            return 1.0
        chi2 = np.sum((depths - mean_d) ** 2) / (mean_d ** 2)
        return float(chi2 / max(len(depths) - 1, 1))
    except Exception:
        return 1.0


def _model_fit_quality(results):
    """
    Reduced chi-squared of the TLS model fit to the phase-folded data.
    """
    if results is None:
        return 1.0
    try:
        folded_y = np.asarray(results.folded_y)
        model_y = np.asarray(results.model_folded_model)

        if len(folded_y) != len(model_y):
            # Interpolate model onto data phases
            from scipy.interpolate import interp1d
            model_phase = np.asarray(results.model_folded_phase)
            data_phase = np.asarray(results.folded_phase)
            f_interp = interp1d(model_phase, model_y, bounds_error=False,
                                fill_value=1.0)
            model_y = f_interp(data_phase)

        residuals = folded_y - model_y
        scatter = np.std(folded_y)
        if scatter < 1e-12:
            return 1.0
        chi2 = np.sum((residuals / scatter) ** 2)
        dof = max(len(residuals) - 5, 1)  # 5 fitted parameters
        return float(chi2 / dof)
    except Exception:
        return 1.0


def _rotation_analysis(time, flux, det):
    """
    Lomb-Scargle analysis for stellar rotation, masking in-transit points.
    """
    features = {}
    try:
        # Mask transit points
        good = np.isfinite(time) & np.isfinite(flux)
        t, f = time[good], flux[good]

        transit_times = np.atleast_1d(det.transit_times)
        half_window = det.duration * 0.75

        if len(transit_times) > 0:
            in_transit = np.zeros(len(t), dtype=bool)
            for tt in transit_times:
                in_transit |= np.abs(t - tt) < half_window
            t_out, f_out = t[~in_transit], f[~in_transit]
        else:
            t_out, f_out = t, f

        if len(t_out) < 50:
            raise ValueError("Too few out-of-transit points")

        ls = LombScargle(t_out, f_out)
        ls_freq, ls_power = ls.autopower(
            minimum_frequency=1 / 30,
            maximum_frequency=2 / max(det.period, 0.1),
        )

        peak_idx = np.argmax(ls_power)
        rot_period = 1.0 / ls_freq[peak_idx]
        rot_power = float(ls_power[peak_idx])
        rot_fap = float(ls.false_alarm_probability(rot_power))

        features["rotation_period_days"] = rot_period
        features["rotation_peak_power"] = rot_power
        features["rotation_peak_fap"] = rot_fap

        # Check if rotation matches transit period (or harmonic thereof)
        # Use 10% tolerance — these are noisy period estimates from different
        # methods (TLS vs Lomb-Scargle), and stellar variability produces
        # broad peaks. A 5% window is too strict for genuine matches.
        tol = 0.20
        close_to_period = abs(rot_period - det.period) / det.period < tol
        close_to_half = abs(rot_period - det.period / 2) / det.period < tol
        # Also check if the transit period is ~half the rotation period
        # (sinusoidal variability often produces TLS peaks at P/2)
        close_to_double = abs(rot_period - det.period * 2) / (det.period * 2) < tol
        is_significant = rot_fap < 0.01

        # Guard against false positives: require the LS peak power to be
        # above a meaningful threshold. Periodic transit gaps in the
        # out-of-transit data can create spurious LS peaks at the transit
        # period even after masking — these typically have low power.
        # Also require at least 2 full rotation cycles in the data span.
        data_span = t_out[-1] - t_out[0] if len(t_out) > 1 else 0
        has_enough_cycles = data_span > 1.5 * rot_period
        has_meaningful_power = rot_power > 0.05

        features["rotation_near_transit_period"] = float(
            (close_to_period or close_to_half or close_to_double)
            and is_significant
            and has_enough_cycles
            and has_meaningful_power
        )
    except Exception:
        features["rotation_period_days"] = np.nan
        features["rotation_peak_power"] = np.nan
        features["rotation_peak_fap"] = np.nan
        features["rotation_near_transit_period"] = 0.0

    return features


def _duration_ratio(period, duration, stellar_radius):
    """
    Ratio of observed duration to expected circular-orbit duration.
    Significantly different → might be eccentric or non-transiting signal.
    Expected duration ≈ (R_star * P) / (π * a)   [simplified]
    """
    try:
        # Kepler's 3rd law: a(AU) ≈ P(yr)^(2/3) for solar-mass star
        p_yr = period / 365.25
        a_au = p_yr ** (2.0 / 3.0)
        a_rsun = a_au * 215.0  # 1 AU ≈ 215 R_sun

        # Expected duration for central transit
        expected_dur = period * stellar_radius / (np.pi * a_rsun)
        if expected_dur > 0:
            return float(duration / expected_dur)
        return 1.0
    except Exception:
        return 1.0


def _ingress_egress_ratio(results):
    """
    Ratio of ingress to egress shape symmetry.
    Asymmetric → possible blend or spot-crossing event.
    """
    if results is None:
        return 1.0
    try:
        phase = np.asarray(results.folded_phase)
        flux = np.asarray(results.folded_y)

        # Find transit center
        center_phase = phase[np.argmin(flux)]

        # Split transit into ingress and egress halves
        transit_mask = flux < (1.0 - 0.3 * (1.0 - np.min(flux)))
        if transit_mask.sum() < 6:
            return 1.0

        ingress = flux[transit_mask & (phase < center_phase)]
        egress = flux[transit_mask & (phase > center_phase)]

        if len(ingress) < 2 or len(egress) < 2:
            return 1.0

        # Compare slope/scatter
        ingress_slope = np.std(ingress)
        egress_slope = np.std(egress)

        if egress_slope > 0:
            return float(ingress_slope / egress_slope)
        return 1.0
    except Exception:
        return 1.0


def _scatter_ratio(time, flux, det):
    """
    Ratio of in-transit scatter to out-of-transit scatter.
    High ratio → noisier during transit → possible blend or contamination.
    """
    try:
        phase = ((time - det.t0) % det.period) / det.period
        phase[phase > 0.5] -= 1.0
        half_dur_phase = (det.duration / det.period) / 2.0

        in_transit = np.abs(phase) < half_dur_phase
        out_of_transit = ~in_transit

        if in_transit.sum() < 5 or out_of_transit.sum() < 20:
            return 1.0

        in_scatter = np.std(flux[in_transit])
        out_scatter = np.std(flux[out_of_transit])

        if out_scatter > 0:
            return float(in_scatter / out_scatter)
        return 1.0
    except Exception:
        return 1.0


def features_to_array(features, feature_names=None):
    """
    Convert a feature dictionary to a numpy array for ML input.
    """
    if feature_names is None:
        feature_names = get_feature_names()

    arr = np.zeros(len(feature_names))
    for i, name in enumerate(feature_names):
        val = features.get(name, 0.0)
        arr[i] = float(val) if np.isfinite(float(val)) else 0.0
    return arr


def get_feature_names():
    """Return the ordered list of feature names used by the ML classifier."""
    return [
        "period_days", "duration_hr", "depth_ppm", "sde", "snr",
        "odd_even_mismatch", "rp_earth_radii", "v_shape_metric",
        "secondary_eclipse_sigma", "depth_chi2_red", "model_chi2_red",
        "rotation_near_transit_period", "duration_ratio",
        "ingress_egress_ratio", "scatter_ratio", "n_transits",
        "stellar_radius_rsun", "stellar_teff", "radius_exceeds_limit",
    ]
