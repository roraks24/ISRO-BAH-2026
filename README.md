<div align="center">

# 🪐 AI-Enabled Exoplanet Transit Detection Pipeline
### Robust Identification, Classification, and MCMC Characterization from Noisy Astronomical Light Curves

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![ISRO BAH 2026](https://img.shields.io/badge/ISRO%20BAH-Challenge%2007-FF9933?style=for-the-badge)](https://www.isro.gov.in/)
[![TESS Mission](https://img.shields.io/badge/Data-NASA%20TESS-0B3D91?style=for-the-badge)](https://tess.mit.edu/)
[![TLS Detection](https://img.shields.io/badge/Algorithm-Transit%20Least%20Squares-4A154B?style=for-the-badge)](https://github.com/hippke/tls)
[![batman Modeling](https://img.shields.io/badge/Transit%20Model-batman-green?style=for-the-badge)](https://lkreidberg.github.io/batman/)
[![MCMC](https://img.shields.io/badge/Posterior-emcee%20MCMC-critical?style=for-the-badge)](https://emcee.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

<br />

**Developed for the ISRO Bharatiya Antariksh Hackathon 2026 (BAH-2026)**  
*Challenge 07: AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves*  
**Team Astroseekers**

[🔬 Pipeline Architecture](#-end-to-end-architecture) · [📊 Diagnostic Gallery](#-validation--diagnostic-gallery) · [📈 Benchmark Results](#-benchmark-validation) · [⚡ Quick Start](#-quick-start)

</div>

---

## 🌌 Overview

Space telescopes like **NASA's Transiting Exoplanet Survey Satellite (TESS)** observe hundreds of thousands of stellar light curves. Uncovering exoplanetary transits within this vast data volume is confounded by instrumental trends, stellar rotation, starspots, and astronomical false positives such as eclipsing binaries (EBs) and background blended sources.

This repository provides an automated, publication-ready exoplanet transit discovery and characterization engine:
- **Optimal Transit Search**: Uses **Transit Least Squares (TLS)** with physical limb-darkening templates, offering ~10% higher sensitivity than traditional Box Least Squares (BLS).
- **Transit-Preserving Detrending**: Employs two-pass Tukey biweight filtering with transit masking to strip stellar variability without truncating transit depths.
- **5-Class Hybrid Classifier**: Combines a Random Forest classifier (200 trees, balanced weights, trained on NASA TOI dispositions) with physics-grounded rule overrides across 19 diagnostic features.
- **Physical Transit Modeling & Bayesian MCMC**: Evaluates realistic light curves via `batman` and samples posterior distributions using `emcee` to obtain rigorous $1\sigma$ parameter uncertainties.
- **Iterative Multi-Planet Recovery**: Sequentially detects, models, masks, and subtracts signals to isolate multi-planet architectures in complex systems.

---

## 🏗️ End-to-End Architecture

```mermaid
flowchart TD
    A[MAST Archive / TESS SPOC 2-min Cadence] --> B[Data Loader & Outlier Rejection]
    B --> C[Two-Pass Tukey Biweight Detrending]
    C --> D[Transit Least Squares Search]
    D --> E{SDE >= 7.0 & SNR >= 5.0?}
    E -->|No| F[Flagged: Non-Detection / Noise Floor]
    E -->|Yes| G[19 Diagnostic Features Extractor]
    G --> H[Hybrid Classifier Engine]
    subgraph Classification Engine
        H --> H1[Random Forest 200 Trees]
        H --> H2[Astrophysical Rule Overrides]
        H1 & H2 --> H3[Final Class & Confidence Score]
    end
    H3 --> I[Physical Model Fitting: batman]
    I --> J[L-BFGS-B Fast Fit]
    I --> K[emcee MCMC Bayesian Posterior Sampling]
    K --> L[7-Panel Publication Diagnostic Sheets]
    L --> M{Iterative Multi-Planet Search?}
    M -->|Mask In-Transit Points| D
    M -->|Complete| N[Detection Catalog CSV & Summary Reports]
```

---

## 📊 Validation & Diagnostic Gallery

The pipeline generates comprehensive 7-panel diagnostic reports for every candidate detection, including phase folds, TLS periodograms, odd-even comparisons, secondary eclipse audits, fit residuals, and diagnostic summaries.

### 1. Real TESS Exoplanet Discovery: WASP-43 b
Validation against confirmed ultra-short-period hot Jupiter **WASP-43 b** using real TESS photometric observations:
![WASP-43 b Diagnostic Sheet](report/fig_wasp43_diagnostic.png)

> **Result:** Detected period $P = 0.8133\,\text{d}$ (Literature: $0.8135\,\text{d}$, error $< 0.02\%$). Radius ratio $R_p/R_\star = 0.1314$. Correctly classified as **`massive_planet`** with **97/100 Confidence** and **100/100 Detection Quality Index (DQI)**.

---

### 2. Shallow Planetary Candidate: Synthetic Benchmark
High-fidelity recovery of a sub-Neptune / Super-Earth candidate in noisy light curves (250 ppm Gaussian noise):
![Planet Diagnostic Sheet](report/fig_planet_diagnostic.png)

> **Result:** Recovered period $P = 6.2709\,\text{d}$ (Injected: $6.2680\,\text{d}$, error $< 0.05\%$). Clean flat-bottomed ingress/egress with $\chi^2_{\text{red}} = 1.006$.

---

### 3. Iterative Multi-Planet System Deconfliction
Iterative detection and signal subtraction isolating multiple transiting exoplanets orbiting the same host star:
![Multi-Planet Recovery](report/fig_multi_planet.png)

> **Result:** Both Planet b ($P = 5.50\,\text{d}$, depth $625\,\text{ppm}$) and Planet c ($P = 9.10\,\text{d}$, depth $324\,\text{ppm}$) successfully disentangled and classified as **`planet_candidate`** with high confidence.

---

### 4. Astrophysical False Positive Rejection: Eclipsing Binary
Robust identification of stellar binary companions through odd-even depth asymmetry and deep V-shape profiles:
![Eclipsing Binary Diagnostic](report/fig_eb_diagnostic.png)

> **Result:** Correctly identified as **`eclipsing_binary`** via significant secondary eclipse depth ($> 5\sigma$) and V-shaped geometry.

---

## 📈 Benchmark Validation

### Synthetic Benchmark Suite (250 ppm noise baseline)
| Target Scenario | Injected Signal | Detected Period | Inferred $R_p/R_\star$ | $\chi^2_{\text{red}}$ | Pipeline Classification | Confidence |
|---|---|---|---|---|---|:---:|
| **Pi Men c Analog** | Exoplanet ($P = 6.268\,\text{d}$) | $6.2709\,\text{d}$ | 0.01559 | 1.006 | ✅ `planet_candidate` | **Medium (62)** |
| **Synthetic EB** | Binary ($P = 3.140\,\text{d}$) | $1.5705\,\text{d}$ | 0.42619 | 87.90 | ✅ `eclipsing_binary` | **Low (10)** |
| **Starspot Modulation** | Stellar Rot. ($12.0\,\text{d}$) | $5.3216\,\text{d}$ | — | — | ✅ `starspot` | **Low (0)** |

### Multi-Planet Injection Test (54-Day Baseline, 200 ppm noise)
| Planet | Injected Period | Detected Period | Period Error | Injected Depth | Inferred Depth | Status |
|---|---|---|---|---|---|:---:|
| **Candidate b** | $5.5000\,\text{d}$ | $5.4988\,\text{d}$ | **0.02%** | $625\,\text{ppm}$ | $631\,\text{ppm}$ | ✅ Confirmed |
| **Candidate c** | $9.1000\,\text{d}$ | $9.1032\,\text{d}$ | **0.03%** | $324\,\text{ppm}$ | $318\,\text{ppm}$ | ✅ Confirmed |

---

## 🔬 Classification & Feature Engineering

The pipeline extracts **19 astrophysical and diagnostic features** grouped into 5 domains:

| Category | Diagnostic Features | Astrophysical Purpose |
|---|---|---|
| **Transit Geometry** | `depth`, `duration`, `period`, `Rp_Rs`, `v_shape_metric`, `ingress_ratio` | Discern flat-bottom planetary transits from grazing/V-shape stellar eclipses |
| **Binarity Diagnostics** | `odd_even_mismatch_sigma`, `secondary_eclipse_depth`, `secondary_sigma` | Detect secondary occultations and alternating primary/secondary eclipse depths |
| **Physical Plausibility** | `implied_planet_radius_earth`, `duration_ratio_vs_circular` | Reject unphysical radii ($R_p > 22\,R_\oplus$) and verify Keplerian orbit compatibility |
| **Statistical Quality** | `SDE`, `SNR`, `reduced_chi2`, `model_fit_quality`, `scatter_ratio` | Quantify signal detection significance against red and white noise floors |
| **Stellar Activity** | `rotation_period_LS`, `rotation_transit_period_match`, `fap_LS` | Prevent stellar rotational modulation and spots from triggering false alarms |

### Astrophysical Classification Schema
```
                      ┌───────────────────────────────────────┐
                      │    TLS Transit Detection (SDE ≥ 7)    │
                      └──────────────────┬────────────────────┘
                                         ▼
                 ┌─────────────────────────────────────────────────┐
                 │       Astrophysical Rule-Based Disambiguation   │
                 └──────┬───────────────┬──────────────┬───────────┘
                        │               │              │
       Odd-Even > 3σ or │   LS Rotation │    All Physical Checks
      Sec. Eclipse > 5σ │    Period Match│         Satisfied
                        ▼               ▼              ▼
               [ECLIPSING BINARY]   [STARSPOT]   [PLANET CANDIDATE]
                                                 ├── Rp < 11.2 R⊕  → Planet Candidate
                                                 └── Rp ≥ 11.2 R⊕  → Massive Planet
```

---

## ⚡ Quick Start

### 1. Prerequisites & Environment Setup
Python 3.10 or higher is recommended. Create an isolated virtual environment:

```bash
# Clone the repository
git clone https://github.com/roraks24/ISRO-BAH-2026.git
cd ISRO-BAH-2026

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Single-Target Analysis
Analyze a specific TESS target by name or TIC identifier:
```bash
python -m src.pipeline --target "Pi Men" --mode single
```

### 3. Run with MCMC Uncertainty Sampling
Execute full Bayesian MCMC sampling (32 walkers, 3000 steps) for orbital inclination, $R_p/R_\star$, and limb darkening:
```bash
python -m src.pipeline --target "WASP-43" --mode single --mcmc
```

### 4. Batch Sector Processing
Process multiple targets across an entire TESS sector:
```bash
python -m src.pipeline --mode batch --sector 1 --max-targets 200
```

### 5. Train the Machine Learning Classifier
Train or fine-tune the Random Forest classifier against NASA's TOI catalog:
```bash
python -m src.pipeline --mode train --toi-catalog --max-targets 500
```

---

## 🎛️ Command-Line Interface (CLI) Reference

```text
python -m src.pipeline [OPTIONS]

Execution Modes:
  --mode single          Analyze a single target (default)
  --mode batch           Batch process targets across sectors or target files
  --mode train           Train Random Forest classifier on catalog dispositions

Target Configuration:
  --target NAME          Target astronomical name or TIC ID (default: "Pi Men")
  --sector N             Specific TESS sector to process (batch mode)
  --targets-file PATH    Path to newline-delimited target names file
  --max-targets N        Cap maximum targets processed in batch
  --max-sectors N        Max sectors to stitch per target (default: 5)

Fitting & Detection Options:
  --sde-threshold F      Minimum TLS Signal Detection Efficiency (default: 7.0)
  --mcmc                 Enable emcee MCMC posterior sampling for uncertainties
  --model-path PATH      Custom path for trained classifier joblib model
  --curated-csv PATH     CSV dataset path for labeled training
  --toi-catalog          Automatically pull and train on NASA TOI catalog
```

---

## 📂 Repository Structure

```
ISRO-BAH-2026/
├── exoplanet_detection_pipeline.py    # Simplified standalone entrypoint
├── multi_planet_demo.py               # Multi-planet detection demonstration
├── synthetic_test.py                  # Synthetic injection & benchmark validation
├── signal_classifier.py               # Standalone heuristic classifier
├── requirements.txt                   # Production dependency manifest
├── src/                               # Modular core library
│   ├── data_loader.py                 # TESS light curve download & MAST queries
│   ├── detrending.py                  # Two-pass Tukey biweight detrending
│   ├── detector.py                    # TLS & BLS multi-planet transit detector
│   ├── feature_extractor.py           # 19-parameter diagnostic feature extractor
│   ├── classifier.py                  # Random Forest + rule override classifier
│   ├── transit_fitter.py              # batman model fitting & emcee MCMC
│   ├── visualizer.py                  # 7-panel diagnostic plot generator
│   ├── report_generator.py            # CSV catalogs & statistical reporting
│   ├── config.py                      # Pipeline constants and hyperparameters
│   └── pipeline.py                    # Master CLI orchestrator
├── report/                            # Comprehensive research reports & figures
│   ├── fig_planet_diagnostic.png      # 7-panel diagnostic sheet: Planet candidate
│   ├── fig_wasp43_diagnostic.png      # 7-panel diagnostic sheet: WASP-43 b
│   ├── fig_multi_planet.png           # Multi-planet deconfusion plot
│   ├── fig_eb_diagnostic.png          # Eclipsing binary diagnostic sheet
│   ├── fig_starspot_diagnostic.png    # Stellar activity / spot diagnostic sheet
│   ├── methodology_report.md          # Formal scientific methodology paper
│   └── report.md                      # Complete challenge summary report
└── results/                           # Generated outputs (auto-created)
    ├── plots/                         # High-resolution diagnostic figures
    └── detection_results_latest.csv   # Comprehensive candidate catalog
```

---

## 📚 Core Scientific References

1. **Hippke, M., & Heller, R. (2019).** *Optimized transit detection algorithm to search for periodic transits of small planets.* Astronomy & Astrophysics, 623, A39.
2. **Kreidberg, L. (2015).** *batman: BAsic Transit Model cAlculatioN in Python.* Publications of the Astronomical Society of the Pacific, 127(957), 1161.
3. **Foreman-Mackey, D., et al. (2013).** *emcee: The MCMC Hammer.* Publications of the Astronomical Society of the Pacific, 125(925), 306.
4. **Kipping, D. M. (2013).** *Efficient, uninformative sampling of limb darkening coefficients for two-parameter laws.* MNRAS, 435(3), 2152–2160.
5. **Ricker, G. R., et al. (2015).** *Transiting Exoplanet Survey Satellite (TESS).* Journal of Astronomical Telescopes, Instruments, and Systems, 1(1), 014003.

---

## 🏆 Attribution & Acknowledgments

Developed by **Team Astroseekers** ([roraks24](https://github.com/roraks24)) for the **ISRO Bharatiya Antariksh Hackathon 2026**.  
Built using astronomical data curated by NASA's Transiting Exoplanet Survey Satellite (TESS) and hosted at the Mikulski Archive for Space Telescopes (MAST).
