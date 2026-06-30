# AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves

**ISRO Bharatiya Antariksh Hackathon 2026 — Challenge 07**
**Team Astroseekers**

---

## 1. Methodology

### 1.1 Data Acquisition and Preprocessing

Light curves are obtained from the TESS mission via the MAST archive using the `lightkurve` library. We download SPOC-processed 2-minute cadence light curves and stitch multiple sectors to maximise phase coverage.

**Preprocessing pipeline:**
1. **NaN/Inf removal**: All non-finite data points are removed.
2. **Outlier rejection**: Iterative 5σ clipping removes cosmic rays and instrumental glitches.
3. **Normalisation**: Flux is normalised to unit median.
4. **Detrending**: A Tukey biweight time-windowed filter (window = 0.5 days) removes slow stellar variability while preserving sharp transit dips. This is superior to Savitzky-Golay filters for transit-like signals, as the biweight estimator is robust to outliers (including transits themselves) and does not "fill in" transit dips.

### 1.2 Transit Detection

We employ **Transit Least Squares (TLS)** (Hippke & Heller, 2019) as the primary detection algorithm. Unlike Box Least Squares (BLS), TLS uses physically realistic transit shapes with stellar limb darkening, yielding ~10% better detection efficiency for small planets.

**Detection criteria:**
- **Signal Detection Efficiency (SDE) ≥ 7**: Standard threshold corresponding to ~1% false positive rate.
- **Signal-to-Noise Ratio (SNR)**: Computed independently as transit depth divided by (out-of-transit scatter / √N_in-transit).
- **Multi-planet search**: After detecting the strongest signal, we mask it and re-run TLS to find additional planets (up to 3 iterations).

### 1.3 Feature Extraction

We extract 19 diagnostic features from each detection to distinguish true transits from false positives:

| Feature Category | Features |
|---|---|
| **Transit geometry** | depth, duration, period, Rp/Rs, V-shape metric, ingress/egress ratio |
| **EB diagnostics** | odd-even depth mismatch, secondary eclipse depth & significance |
| **Physical plausibility** | implied planet radius (R⊕), duration ratio vs. expected circular orbit |
| **Statistical quality** | SDE, SNR, reduced χ², model fit quality, scatter ratio |
| **Stellar activity** | rotation period (Lomb-Scargle), rotation-transit period match |
| **Metadata** | stellar radius, Teff, number of observed transits |

### 1.4 Signal Classification

**Hybrid ML + rule-based classifier** with five classes:
- **planet_candidate**: Genuine exoplanet transit
- **eclipsing_binary**: Stellar companion eclipses
- **blend**: Diluted background source
- **starspot**: Stellar rotation modulation
- **false_alarm**: Instrumental noise or artifact

**ML component**: A Random Forest (200 estimators, balanced class weights) trained on NASA TOI catalog dispositions with 5-fold stratified cross-validation.

**Rule-based overrides**: Extreme cases (e.g., implied radius > 22 R⊕, odd-even mismatch > 3σ, secondary eclipse > 5σ) trigger deterministic overrides regardless of ML output, ensuring no physically impossible classifications pass through.

### 1.5 Transit Model Fitting

Physical transit models are generated using the `batman` package (Kreidberg, 2015) with quadratic limb darkening.

**Two-tier fitting strategy:**
1. **Fast fit** (all detections): `scipy.optimize.minimize` (L-BFGS-B) fits T₀, Rp/Rs, a/Rs, inclination, and limb darkening coefficients. Runtime: ~1-5 seconds per target.
2. **MCMC fit** (high-confidence candidates): `emcee` Ensemble Sampler (32 walkers, 3000 steps, 500 burn-in) samples the full posterior distribution. Provides proper 1σ uncertainties from the 16th/84th percentiles of the marginalised posteriors.

**Fitted parameters:**
- Orbital period P (days)
- Transit epoch T₀ (BJD)
- Planet-to-star radius ratio Rp/Rs
- Scaled semi-major axis a/Rs
- Orbital inclination i (degrees)
- Impact parameter b
- Quadratic limb darkening coefficients (u₁, u₂)

---

## 2. Assumptions

1. **Circular orbits**: Eccentricity is fixed at e = 0. Justified for short-period planets (P < 20 d) where tidal circularisation is efficient.
2. **Single-star systems**: Stellar parameters (radius, Teff) are taken directly from the TESS Input Catalog (TIC). Unresolved binarity may introduce systematic errors in Rp estimates.
3. **Quadratic limb darkening**: A two-parameter limb darkening law is assumed. This is standard for TESS-band photometry and adequate for the precision of 2-minute cadence data.
4. **No stellar variability model**: Stellar variability is handled by detrending rather than simultaneous fitting. This is justified for the biweight filter which is robust to transit-like dips.
5. **Threshold-based detection**: SDE ≥ 7 is used as the detection threshold, following the community standard (Hippke & Heller, 2019).

---

## 3. Tools and Libraries

| Library | Version | Purpose |
|---|---|---|
| `lightkurve` | ≥ 2.4 | TESS data download, stitching, normalisation |
| `transitleastsquares` | ≥ 1.0.31 | Transit detection with realistic models |
| `batman-package` | ≥ 2.4 | Physical transit light curve generation |
| `emcee` | ≥ 3.1 | MCMC posterior sampling (Foreman-Mackey et al., 2013) |
| `scikit-learn` | ≥ 1.3 | Random Forest classifier, cross-validation |
| `astropy` | ≥ 5.3 | Lomb-Scargle periodograms, BLS, time handling |
| `astroquery` | ≥ 0.4 | TIC catalog queries, MAST archive access |
| `scipy` | ≥ 1.11 | L-BFGS-B optimisation, signal filtering |
| `numpy` / `pandas` | standard | Array operations, data management |
| `matplotlib` / `seaborn` | standard | Publication-quality visualisation |
| `corner` | ≥ 2.2 | MCMC posterior corner plots |

---

## 4. Uncertainty Estimation

### 4.1 Transit Parameter Uncertainties

Parameter uncertainties are estimated via **MCMC posterior sampling** using `emcee`:

- **Prior distributions**: Uniform priors within physically motivated bounds (e.g., 0 < Rp/Rs < 0.5, 70° < i < 90°). Limb darkening priors follow the Kipping (2013) parameterisation constraint u₁ + u₂ ∈ [0, 1].
- **Likelihood**: Gaussian likelihood assuming independent, identically distributed photometric errors.
- **Convergence**: 32 walkers × 3000 steps with 500-step burn-in. Convergence is assessed by visual inspection of chains and the Gelman-Rubin statistic.
- **Reported uncertainties**: Median values with 1σ intervals from the 16th and 84th percentiles of the marginalised posterior distributions.

### 4.2 Classification Confidence

- **ML classifier**: Class probabilities from the Random Forest (fraction of trees voting for each class).
- **Rule-based classifier**: A transparent heuristic score (0–100) based on the TLS SDE, penalised by each false-positive flag that triggers (odd-even mismatch, secondary eclipse, radius ceiling, rotation match). The penalty magnitudes are proportional to how badly each diagnostic was failed.
- **Confidence levels**: High (≥ 80%), Medium (50–79%), Low (< 50%).

### 4.3 Detection Significance

- **SDE**: Signal Detection Efficiency from TLS — measures how far the best-fit period stands above the periodogram noise floor. SDE ≥ 7 corresponds to ~1% false positive rate.
- **SNR**: Independent signal-to-noise ratio computed as depth / (scatter / √N_in_transit).
- **Combined significance**: Weighted combination of SDE (40%), SNR (40%), and number of observed transits (20%).

---

## References

1. Hippke, M. & Heller, R. (2019). "Optimized transit detection algorithm to search for periodic transits of small planets." *A&A*, 623, A39.
2. Kreidberg, L. (2015). "batman: BAsic Transit Model cAlculatioN in Python." *PASP*, 127, 1161.
3. Foreman-Mackey, D. et al. (2013). "emcee: The MCMC Hammer." *PASP*, 125, 306.
4. Kipping, D. M. (2013). "Efficient, uninformative sampling of limb darkening coefficients." *MNRAS*, 435, 2152.
5. Ricker, G. R. et al. (2015). "Transiting Exoplanet Survey Satellite (TESS)." *JATIS*, 1, 014003.
