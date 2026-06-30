"""
transit_fitter.py — Physical transit model fitting with batman + MCMC.

Two-tier approach:
  1. Fast fit (scipy.optimize) — for all detections (~1-5 sec)
  2. MCMC fit (emcee) — for top candidates only (~30-120 sec)

Parameters estimated:
  - Orbital period (days)
  - Transit epoch T0 (BJD)
  - Planet-to-star radius ratio Rp/Rs
  - Scaled semi-major axis a/Rs
  - Orbital inclination i (degrees)
  - Limb darkening coefficients (u1, u2)
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


try:
    import batman
    HAS_BATMAN = True
except ImportError:
    HAS_BATMAN = False

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False


@dataclass
class FitResult:
    """Container for transit fit results."""
    # Best-fit parameters
    period: float = 0.0
    t0: float = 0.0
    rp_rs: float = 0.0
    a_rs: float = 0.0
    inclination: float = 90.0
    u1: float = 0.3
    u2: float = 0.1
    # Derived quantities
    depth_ppm: float = 0.0
    duration_hr: float = 0.0
    impact_parameter: float = 0.0
    # Uncertainties (from MCMC)
    period_err: float = 0.0
    t0_err: float = 0.0
    rp_rs_err: float = 0.0
    a_rs_err: float = 0.0
    inclination_err: float = 0.0
    depth_ppm_err: float = 0.0
    duration_hr_err: float = 0.0
    # Fit quality
    chi2_red: float = 0.0
    bic: float = 0.0
    # MCMC samples (for corner plots)
    mcmc_samples: Optional[np.ndarray] = None
    mcmc_labels: list = field(default_factory=list)
    method: str = "scipy"
    converged: bool = False
    fit_quality: str = "Good"


def generate_transit_model(time, period, t0, rp_rs, a_rs, inclination,
                           u1=0.3, u2=0.1, ecc=0.0, omega=90.0):
    """
    Generate a synthetic transit light curve using batman.

    Returns flux array (baseline = 1.0, transit dip < 1.0).
    """
    if not HAS_BATMAN:
        # Fallback: simple box model
        return _box_transit_model(time, period, t0, rp_rs)

    params = batman.TransitParams()
    params.t0 = t0
    params.per = period
    params.rp = rp_rs
    params.a = a_rs
    params.inc = inclination
    params.ecc = ecc
    params.w = omega
    params.u = [u1, u2]
    params.limb_dark = "quadratic"

    m = batman.TransitModel(params, time)
    return m.light_curve(params)


def _box_transit_model(time, period, t0, rp_rs):
    """Simple box transit model fallback when batman is unavailable."""
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    depth = rp_rs ** 2
    duration_phase = 0.02  # approximate
    flux = np.ones_like(time)
    in_transit = np.abs(phase) < duration_phase / 2
    flux[in_transit] = 1.0 - depth
    return flux


# ═══════════════════════════════════════════════════════════════════════
#  FAST FIT (scipy.optimize)
# ═══════════════════════════════════════════════════════════════════════

def fit_transit_fast(time, flux, flux_err=None, initial_params=None,
                     detection=None):
    """
    Fast transit model fit using scipy.optimize.minimize.

    Parameters
    ----------
    time, flux : arrays
    flux_err : array or None (defaults to scatter of out-of-transit data)
    initial_params : dict with keys: period, t0, rp_rs, a_rs, inclination
    detection : DetectionResult (used for initial params if initial_params is None)

    Returns
    -------
    FitResult
    """
    # Set up initial parameters
    if initial_params is None and detection is not None:
        initial_params = {
            "period": detection.period,
            "t0": detection.t0,
            "rp_rs": detection.rp_rs,
            "a_rs": _estimate_a_rs(detection.period, detection.duration),
            "inclination": 89.0,
        }
    elif initial_params is None:
        raise ValueError("Must provide either initial_params or detection.")

    p0 = initial_params

    # Estimate flux errors if not provided
    if flux_err is None:
        phase = ((time - p0["t0"]) % p0["period"]) / p0["period"]
        phase[phase > 0.5] -= 1.0
        oot = np.abs(phase) > 0.1
        flux_err = np.full_like(flux, np.std(flux[oot]) if oot.sum() > 10 else np.std(flux))

    # Parameter vector: [t0, rp_rs, a_rs, inclination, u1, u2]
    # Period is held fixed at TLS value (very well constrained)
    period_fixed = p0["period"]
    x0 = np.array([
        p0["t0"],
        max(p0["rp_rs"], 0.001),
        max(p0.get("a_rs", 15.0), 2.0),
        p0.get("inclination", 89.0),
        0.3,   # u1
        0.1,   # u2
    ])

    # Bounds
    bounds = [
        (p0["t0"] - 0.1, p0["t0"] + 0.1),    # t0
        (0.001, 0.5),                          # rp_rs
        (1.5, 200.0),                          # a_rs
        (70.0, 90.0),                          # inclination
        (0.0, 1.0),                            # u1
        (-0.5, 1.0),                           # u2
    ]

    def neg_log_like(params):
        t0, rp_rs, a_rs, inc, u1, u2 = params
        try:
            model = generate_transit_model(
                time, period_fixed, t0, rp_rs, a_rs, inc, u1, u2
            )
            residuals = (flux - model) / flux_err
            return 0.5 * np.sum(residuals ** 2)
        except Exception:
            return 1e10

    result = minimize(
        neg_log_like, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 1000},
    )

    popt = result.x
    t0_fit, rp_rs_fit, a_rs_fit, inc_fit, u1_fit, u2_fit = popt

    # Check if any fitted parameters hit their bounds
    converged = result.success
    bounds_lower = [b[0] for b in bounds]
    bounds_upper = [b[1] for b in bounds]
    param_names = ['t0', 'rp_rs', 'a_rs', 'inclination', 'u1', 'u2']
    for i, (name, val) in enumerate(zip(param_names, popt)):
        lo, hi = bounds_lower[i], bounds_upper[i]
        bound_range = hi - lo
        if bound_range > 0:
            if (val - lo) / bound_range < 0.01 or (hi - val) / bound_range < 0.01:
                logger.warning(f"Parameter '{name}' = {val:.4f} hit bound [{lo}, {hi}] — fit may not have converged")
                converged = False

    # Compute derived quantities
    depth = rp_rs_fit ** 2
    impact_b = a_rs_fit * np.cos(np.radians(inc_fit))

    # Duration estimate
    try:
        sin_inc = np.sin(np.radians(inc_fit))
        sqrt_arg = (1 + rp_rs_fit) ** 2 - impact_b ** 2
        if sqrt_arg < 0:
            logger.warning(
                f"Non-physical fit: impact_b={impact_b:.3f} > 1+Rp/Rs="
                f"{1+rp_rs_fit:.3f} — duration set to 0"
            )
            dur_days = 0.0
        else:
            dur_days = (period_fixed / np.pi) * np.arcsin(
                (1.0 / a_rs_fit) * np.sqrt(sqrt_arg) / sin_inc
            )
    except Exception:
        dur_days = 0.0

    # Chi-squared
    model_best = generate_transit_model(
        time, period_fixed, t0_fit, rp_rs_fit, a_rs_fit, inc_fit,
        u1_fit, u2_fit
    )
    residuals = (flux - model_best) / flux_err
    chi2 = float(np.sum(residuals ** 2))
    n_params = 6
    dof = max(len(flux) - n_params, 1)
    chi2_red = chi2 / dof
    bic = chi2 + n_params * np.log(len(flux))

    return FitResult(
        period=period_fixed,
        t0=t0_fit,
        rp_rs=rp_rs_fit,
        a_rs=a_rs_fit,
        inclination=inc_fit,
        u1=u1_fit,
        u2=u2_fit,
        depth_ppm=depth * 1e6,
        duration_hr=dur_days * 24,
        impact_parameter=impact_b,
        chi2_red=chi2_red,
        bic=bic,
        method="scipy",
        converged=converged,
        fit_quality=(
            "Good" if converged and chi2_red < 3.0 else
            "Questionable" if converged or chi2_red < 10.0 else
            "Failed"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
#  MCMC FIT (emcee)
# ═══════════════════════════════════════════════════════════════════════

def fit_transit_mcmc(time, flux, flux_err=None, initial_params=None,
                     detection=None, nwalkers=32, nsteps=3000,
                     burn_in=500, show_progress=True):
    """
    MCMC transit fit using emcee for proper uncertainty estimation.

    Returns FitResult with mcmc_samples and parameter uncertainties.
    """
    if not HAS_EMCEE:
        print("emcee not installed — falling back to fast fit.")
        return fit_transit_fast(time, flux, flux_err, initial_params, detection)

    # Start from fast fit
    fast_result = fit_transit_fast(time, flux, flux_err, initial_params, detection)
    period_fixed = fast_result.period

    if flux_err is None:
        phase = ((time - fast_result.t0) % period_fixed) / period_fixed
        phase[phase > 0.5] -= 1.0
        oot = np.abs(phase) > 0.1
        flux_err = np.full_like(flux, np.std(flux[oot]) if oot.sum() > 10 else np.std(flux))

    # Parameter vector: [t0, rp_rs, a_rs, inclination, u1, u2]
    ndim = 6
    
    # Enforce strict physical priors on the initial position to prevent emcee crashing 
    # if fast fit values lie on or outside boundaries.
    t0_init = np.clip(fast_result.t0, fast_result.t0 - 0.09, fast_result.t0 + 0.09)
    rp_rs_init = np.clip(fast_result.rp_rs, 0.002, 0.49)
    a_rs_init = np.clip(fast_result.a_rs, 1.6, 199.0)
    inc_init = np.clip(fast_result.inclination, 70.1, 89.9)
    u1_init = np.clip(fast_result.u1, 0.01, 0.99)
    u2_init = np.clip(fast_result.u2, -0.49, 0.99)
    
    # Ensure Kipping limb darkening constraint is strictly satisfied: 0.0 < u1 + u2 < 1.0
    if u1_init + u2_init >= 1.0:
        excess = (u1_init + u2_init) - 0.99
        u1_init -= excess / 2.0
        u2_init -= excess / 2.0
    elif u1_init + u2_init <= 0.0:
        deficit = 0.01 - (u1_init + u2_init)
        u1_init += deficit / 2.0
        u2_init += deficit / 2.0
        
    p0_vec = np.array([t0_init, rp_rs_init, a_rs_init, inc_init, u1_init, u2_init])

    def log_prior(params):
        t0, rp_rs, a_rs, inc, u1, u2 = params
        if not (p0_vec[0] - 0.1 < t0 < p0_vec[0] + 0.1):
            return -np.inf
        if not (0.001 < rp_rs < 0.5):
            return -np.inf
        if not (1.5 < a_rs < 200.0):
            return -np.inf
        if not (70.0 < inc < 90.0):
            return -np.inf
        if not (0.0 < u1 < 1.0):
            return -np.inf
        if not (-0.5 < u2 < 1.0):
            return -np.inf
        # Kipping (2013) prior on limb darkening
        if u1 + u2 > 1.0 or u1 + u2 < 0.0:
            return -np.inf
        return 0.0

    def log_likelihood(params):
        t0, rp_rs, a_rs, inc, u1, u2 = params
        try:
            model = generate_transit_model(
                time, period_fixed, t0, rp_rs, a_rs, inc, u1, u2
            )
            return -0.5 * np.sum(((flux - model) / flux_err) ** 2)
        except Exception:
            return -np.inf

    def log_probability(params):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        ll = log_likelihood(params)
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    # Initialize walkers around the fast fit solution
    pos = p0_vec + 1e-4 * np.random.randn(nwalkers, ndim)

    # Run MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    print(f"Running MCMC: {nwalkers} walkers × {nsteps} steps...")
    sampler.run_mcmc(pos, nsteps, progress=show_progress)

    # Extract samples after burn-in
    samples = sampler.get_chain(discard=burn_in, flat=True)

    # Median and 1-sigma uncertainties from posterior
    medians = np.median(samples, axis=0)
    lower = np.percentile(samples, 16, axis=0)
    upper = np.percentile(samples, 84, axis=0)
    errors = (upper - lower) / 2.0

    t0_m, rp_rs_m, a_rs_m, inc_m, u1_m, u2_m = medians
    t0_e, rp_rs_e, a_rs_e, inc_e, u1_e, u2_e = errors

    # Derived quantities
    depth = rp_rs_m ** 2
    depth_err = 2 * rp_rs_m * rp_rs_e * 1e6  # error propagation
    impact_b = a_rs_m * np.cos(np.radians(inc_m))

    # Duration
    try:
        sin_inc = np.sin(np.radians(inc_m))
        sqrt_arg = (1 + rp_rs_m) ** 2 - impact_b ** 2
        if sqrt_arg < 0:
            logger.warning(
                f"Non-physical MCMC fit: impact_b={impact_b:.3f} > 1+Rp/Rs="
                f"{1+rp_rs_m:.3f} — duration set to 0"
            )
            dur_days = 0.0
        else:
            dur_days = (period_fixed / np.pi) * np.arcsin(
                (1.0 / a_rs_m) * np.sqrt(sqrt_arg) / sin_inc
            )
    except Exception:
        dur_days = 0.0

    # Chi-squared
    model_best = generate_transit_model(
        time, period_fixed, t0_m, rp_rs_m, a_rs_m, inc_m, u1_m, u2_m
    )
    residuals = (flux - model_best) / flux_err
    chi2 = float(np.sum(residuals ** 2))
    dof = max(len(flux) - ndim, 1)

    labels = ["T0", "Rp/Rs", "a/Rs", "inc", "u1", "u2"]

    return FitResult(
        period=period_fixed,
        t0=t0_m,
        rp_rs=rp_rs_m,
        a_rs=a_rs_m,
        inclination=inc_m,
        u1=u1_m,
        u2=u2_m,
        depth_ppm=depth * 1e6,
        duration_hr=dur_days * 24,
        impact_parameter=impact_b,
        period_err=0.0,  # period held fixed
        t0_err=t0_e,
        rp_rs_err=rp_rs_e,
        a_rs_err=a_rs_e,
        inclination_err=inc_e,
        depth_ppm_err=depth_err,
        duration_hr_err=0.0,
        chi2_red=chi2 / dof,
        bic=chi2 + ndim * np.log(len(flux)),
        mcmc_samples=samples,
        mcmc_labels=labels,
        method="mcmc",
        converged=True,
        fit_quality="Good" if (chi2 / dof) < 3.0 else "Questionable",
    )


# ═══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _estimate_a_rs(period_days, duration_days):
    """
    Rough estimate of a/Rs from the period and duration,
    assuming central transit and circular orbit.
    a/Rs ≈ P / (π * T)
    """
    if duration_days <= 0:
        return 15.0
    a_rs = period_days / (np.pi * duration_days)
    return float(np.clip(a_rs, 2.0, 200.0))


def compute_model_lightcurve(time, fit_result):
    """Generate the model light curve from a FitResult."""
    return generate_transit_model(
        time, fit_result.period, fit_result.t0, fit_result.rp_rs,
        fit_result.a_rs, fit_result.inclination, fit_result.u1, fit_result.u2,
    )


def fold_lightcurve(time, flux, period, t0):
    """Phase-fold a light curve."""
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]


def bin_phase_curve(phase, flux, n_bins=200):
    """Bin a phase-folded light curve for cleaner visualization."""
    bins = np.linspace(phase.min(), phase.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_flux = np.zeros(n_bins)
    bin_err = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        if mask.sum() > 0:
            bin_flux[i] = np.median(flux[mask])
            bin_err[i] = np.std(flux[mask]) / np.sqrt(mask.sum())
        else:
            bin_flux[i] = np.nan
            bin_err[i] = np.nan

    valid = np.isfinite(bin_flux)
    return bin_centers[valid], bin_flux[valid], bin_err[valid]
