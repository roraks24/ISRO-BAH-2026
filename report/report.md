# AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves

**Team Astroseekers — ISRO BAH 2026, Challenge 07**

---

## 1. Introduction & Objective

We present a fully automated, end-to-end pipeline for detecting and classifying exoplanet transit signals in noisy TESS photometric data. Given a raw light curve, the pipeline (i) removes systematic trends while preserving transit morphology, (ii) searches for periodic dips using optimal-matched-filter algorithms, (iii) classifies each detection into astrophysical categories (planet transit, eclipsing binary, stellar activity, or instrumental noise), (iv) fits a physical transit model to estimate orbital and planetary parameters, and (v) produces publication-quality diagnostic visualizations. The system handles multi-planet systems through iterative detect-mask-redetect and operates in both single-target and batch-sector modes.

---

## 2. Methodology

### 2.1 Data Acquisition & Preprocessing

TESS light curves are downloaded from the MAST archive via the `lightkurve` library, prioritizing SPOC 2-minute cadence products. Multi-sector observations are stitched to extend the temporal baseline. Stellar parameters (radius, Teff) are queried from the TESS Input Catalog (TIC) via `astroquery`.

**Cleaning:** NaN/Inf values are removed, flux is median-normalized, and a conservative 20σ iterative sigma-clip removes outliers while preserving genuine deep eclipses.

### 2.2 Transit-Aware Detrending

Systematic trends are removed using a two-pass strategy:

1. **Coarse pass:** A Tukey biweight time-windowed filter (`wotan`, window = 0.5 d) flattens the light curve. A quick TLS search identifies any dominant periodic signal.
2. **Fine pass:** If a credible transit is found (SDE ≥ 7, realistic duration 0.5–6 hr, depth < 1%), those in-transit points are masked (1.5× duration buffer, ≤15% of data) before re-detrending. This prevents the filter from "filling in" the transit and preserving transit depth.

### 2.3 Transit Detection

**Primary method — Transit Least Squares (TLS):** Unlike the traditional Box Least Squares (BLS) which assumes a box-shaped dip, TLS uses physically realistic limb-darkened transit templates, achieving higher sensitivity to shallow planetary transits (Hippke & Heller, 2019). The period search spans 0.5–20 days. A detection is considered significant when:

- **SDE ≥ 7.0** (Signal Detection Efficiency)
- **SNR ≥ 5.0**, computed as: SNR = δ / (σ / √n), where δ is transit depth, σ is out-of-transit scatter, and n is the number of in-transit points.

**Multi-planet search:** After detecting the strongest signal, in-transit points are replaced with baseline flux + Gaussian noise (seeded RNG for reproducibility), and TLS is re-run. This iterates up to 3 planets per target, with a ±2% period duplicate check.

### 2.4 Feature Extraction & Classification

For each detection, 19 diagnostic features are computed:

| Feature Category | Key Features | Purpose |
|---|---|---|
| **Detection metrics** | SDE, SNR, n_transits | Signal significance |
| **Eclipse diagnostics** | Odd-even mismatch (σ), secondary eclipse depth/σ, V-shape metric | EB identification |
| **Physical parameters** | Rp (R⊕), depth (ppm), duration (hr), period (d) | Planet characterization |
| **Stellar activity** | Lomb-Scargle rotation period, peak power, FAP | Starspot identification |
| **Transit morphology** | Duration ratio, ingress/egress symmetry, scatter ratio | Shape analysis |

**Classification** uses a hybrid ML + rule-based approach:

- **Rule-based vetter (primary):** An interpretable decision engine evaluates each feature against calibrated thresholds, accumulating penalty scores. It includes rescue logic for hot Jupiters (whose genuine thermal emission mimics secondary eclipses) and starspot disambiguation (sinusoidal signals producing spurious odd-even mismatches). Priority order: eclipsing binary > blend > starspot > false alarm > massive planet > planet candidate.

- **ML classifier (optional):** A Random Forest (200 trees, balanced class weights, 5-fold stratified CV) trained on the TOI catalog provides probabilistic predictions. The rule-based engine can override the ML verdict when strong false-positive evidence exists (confidence < 20).

**Output categories** (mapped to Challenge-07 format):

| Internal Class | Challenge-07 Class | Diagnostic Triggers |
|---|---|---|
| `planet_candidate` | Planet | Passes all checks, Rp < 11.2 R⊕ |
| `massive_planet` | Massive Planet | Passes all checks, Rp ≥ 11.2 R⊕ (≈1 R_J) |
| `eclipsing_binary` | Eclipsing Binary | Odd-even > 3σ, secondary eclipse > 5σ, V-shape > 0.85, or depth > 5% |
| `starspot` | False Alarm | Dip period matches stellar rotation (LS power > 0.05, FAP < 0.01) |
| `false_alarm` | False Alarm | SDE < 7 or < 2 transits |

**Confidence scoring:** Base score scales linearly with SDE (0–100), penalized by each triggered diagnostic flag (5–40 points each). Final confidence: High (≥70), Medium (≥40), Low (<40).

### 2.5 Transit Model Fitting & Parameter Estimation

A physical transit model is generated using the `batman` package (Kreidberg, 2015) with quadratic limb darkening and circular orbits. Six free parameters are fitted: T₀, Rp/Rs, a/Rs, inclination, u₁, u₂ (period fixed from TLS).

- **Fast fit:** `scipy.optimize.minimize` (L-BFGS-B), ~1–5 sec per target. Convergence is verified by checking that no parameter hits its bound.
- **MCMC fit (optional):** `emcee` ensemble sampler (32 walkers × 3000 steps, 500-step burn-in) with Kipping (2013) limb-darkening priors. Posterior medians provide best-fit values; 16th/84th percentiles yield 1σ uncertainties.
- **Fit quality:** "Good" (converged, χ²_red < 3), "Questionable" (partially converged), or "Failed".

**Derived parameters:**
- Transit depth: (Rp/Rs)² × 10⁶ ppm
- Planet radius: Rp/Rs × R★ × 109.2 R⊕
- Impact parameter: b = (a/Rs) × cos(i)
- Duration: analytical from Winn (2010) formula

### 2.6 Uncertainty Estimation

| Method | Uncertainty Source |
|---|---|
| **TLS detection** | SDE and SNR quantify detection significance; FAP from bootstrap |
| **Fast fit** | Bound-hitting flags; χ²_red indicates goodness of fit |
| **MCMC** | Full posterior distributions → 1σ from 16th/84th percentiles |
| **Classification** | Confidence score (0–100) with interpretable penalty breakdown |
| **DQI** | Detection Quality Index combining SDE, SNR, fit quality, and warnings |

---

## 3. Assumptions

1. **Circular orbits** (eccentricity = 0) — valid for most short-period planets tidally circularized.
2. **Single host star** — no dilution correction for unresolved companions.
3. **Quadratic limb darkening** — standard for TESS bandpass.
4. **Period range 0.5–20 days** — optimized for TESS single-sector baselines (~27 days).
5. **Stellar parameters from TIC** — assumed accurate; fallback to solar values if unavailable.
6. **White noise model** — correlated (red) noise not explicitly modeled in fitting.

---

## 4. Tools & Libraries

| Library | Version | Purpose |
|---|---|---|
| `transitleastsquares` | 1.32 | Optimal transit detection (TLS) |
| `batman-package` | — | Physical transit model generation |
| `emcee` | — | MCMC posterior sampling |
| `wotan` | — | Transit-masked biweight detrending |
| `lightkurve` | — | TESS data download & stitching |
| `astropy` | — | BLS, Lomb-Scargle periodogram, time utilities |
| `astroquery` | — | TIC/MAST catalog queries |
| `scikit-learn` | — | Random Forest classifier, cross-validation |
| `scipy` | — | L-BFGS-B optimization, signal processing |
| `matplotlib` | — | All visualizations |
| `numpy`, `pandas` | — | Numerical computation, data I/O |

---

## 5. Results & Validation

### 5.1 Synthetic Test Suite (3 injected signals, 250 ppm noise)

| Target | True Signal | Detected P (d) | Classification | Confidence | Rp/Rs (fit) | χ²_red |
|---|---|---|---|---|---|---|
| Pi Men c analog | Planet (P=6.268d) | 6.2709 | ✅ planet_candidate | Medium (62) | 0.01559 | 1.006 |
| Synthetic EB | Binary (P=3.14d) | 1.5705 | ✅ eclipsing_binary | Low (10) | 0.42619 | 87.90 |
| Starspot | Rotation (12d) | 5.3216 | ✅ starspot | Low (0) | — | — |

Period recovery error for the planet: <0.05%. Rp/Rs error: ~4%.

### 5.2 Multi-Planet Recovery

Two injected planets (P_b=5.5d, depth=625 ppm; P_c=9.1d, depth=324 ppm) in 200 ppm noise over 54 days. Both recovered with correct periods (error < 0.04%) and classified as `planet_candidate` (High confidence).

### 5.3 Real TESS Target — WASP-43 b

| Parameter | Pipeline Value | Literature Value |
|---|---|---|
| Period | 0.8133 d | 0.8135 d |
| Rp/Rs | 0.1314 | 0.1594 |
| a/Rs | 5.71 | 4.92 |
| Inclination | 89.2° | 82.1° |
| Classification | massive_planet (High, 97/100) | Confirmed hot Jupiter |
| DQI | 100/100 | — |

### 5.4 Visualization

Each candidate produces a 7-panel diagnostic sheet: detrended light curve with transit markers, TLS periodogram, phase-folded transit with best-fit model overlay, odd-vs-even transit comparison, fit residuals, secondary eclipse check, and a summary box with classification verdict, fitted parameters, and diagnostic metrics. Individual transit galleries show up to 12 separate transit events.

---

## 6. Conclusion

Our pipeline achieves robust, automated exoplanet detection and classification from noisy TESS light curves. The transit-aware detrending preserves signal integrity, the TLS algorithm provides optimal sensitivity to limb-darkened transits, and the hybrid rule-based + ML classifier correctly separates planets from eclipsing binaries and stellar activity with interpretable confidence scores. The system successfully recovers known exoplanets (WASP-43 b), handles multi-planet systems, and produces comprehensive diagnostic visualizations — meeting all requirements of Challenge 07.

---

*References: Hippke & Heller (2019, A&A 623, A39); Kreidberg (2015, PASP 127, 1161); Kipping (2013, MNRAS 435, 2152); Winn (2010, arXiv:1001.2010)*
