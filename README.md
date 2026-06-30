# Exoplanet Transit Detection Pipeline

**ISRO BAH 2026 — Challenge 07: AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves**

**Team Astroseekers**

---

## Overview

An AI-driven pipeline that robustly identifies, classifies, and characterises exoplanet transit signals from noisy TESS light curves. The pipeline combines Transit Least Squares (TLS) detection with a Random Forest + rule-based hybrid classifier and physical transit model fitting via `batman` + MCMC.

## Features

- **Transit Detection**: TLS and BLS with multi-planet iterative search
- **Signal Classification**: 5-class hybrid ML + rule-based classifier (planet, EB, blend, starspot, false alarm)
- **Model Fitting**: Physical transit models with batman, parameter uncertainties via MCMC (emcee)
- **Batch Processing**: Full TESS sector analysis (20-30k light curves)
- **Rich Visualization**: 7-panel diagnostic sheets, sector summaries, MCMC corner plots
- **Confidence Metrics**: SDE, SNR, classification probabilities, and posterior uncertainties

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run on a Single Target

```bash
python exoplanet_detection_pipeline.py
```

Or use the CLI:

```bash
python -m src.pipeline --target "Pi Men" --mode single
```

### 3. Run with MCMC Uncertainties

```bash
python -m src.pipeline --target "Pi Men" --mode single --mcmc
```

### 4. Train the ML Classifier

```bash
python -m src.pipeline --mode train --toi-catalog --max-targets 200
```

### 5. Batch Process a Sector

```bash
python -m src.pipeline --mode batch --sector 1 --max-targets 500
```

### 6. Use a Curated Dataset

```bash
python -m src.pipeline --mode train --curated-csv data/labeled_targets.csv
```

## Project Structure

```
ISRO/
├── exoplanet_detection_pipeline.py   # Quick single-target entry point
├── signal_classifier.py              # Legacy rule-based classifier (fallback)
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── src/                              # Core pipeline modules
│   ├── __init__.py
│   ├── data_loader.py                # TESS data download & ingestion
│   ├── detrending.py                 # Light curve detrending (biweight, SavGol)
│   ├── detector.py                   # Transit detection (TLS, BLS, multi-planet)
│   ├── feature_extractor.py          # 19 diagnostic features for classification
│   ├── classifier.py                 # Random Forest + rule-based hybrid
│   ├── transit_fitter.py             # batman model fitting + emcee MCMC
│   ├── visualizer.py                 # Publication-quality plots
│   ├── pipeline.py                   # End-to-end orchestrator (CLI)
│   └── report_generator.py           # CSV export & summary reports
│
├── results/                          # Pipeline output (auto-created)
│   ├── plots/                        # Diagnostic plots per candidate
│   ├── detection_results_latest.csv  # Latest results table
│   └── pipeline_summary_report.txt   # Summary statistics
│
├── models/                           # Trained ML models (auto-created)
│   └── transit_classifier.joblib     # Saved Random Forest model
│
├── data/                             # Cached data (auto-created)
│   └── sector_cache/                 # Cached light curves (.npz)
│
└── report/
    └── methodology_report.md         # 3-page methodology report
```

## Pipeline Architecture

```
TESS Data (MAST) ──→ Data Loader ──→ Detrending (Biweight)
                                          │
                                    Transit Detection (TLS)
                                          │
                                   ┌──────┴──────┐
                                   │  SDE ≥ 7?   │
                                   └──────┬──────┘
                                     Yes  │  No → Skip
                                          │
                                  Feature Extraction (19 features)
                                          │
                              ┌───────────┴───────────┐
                              │  ML Classifier Ready?  │
                              └───────────┬───────────┘
                                Yes │          │ No
                                    │          │
                            Random Forest   Rule-Based
                            + Rule Override  Classifier
                                    │          │
                                    └────┬─────┘
                                         │
                               Transit Model Fitting
                               (batman + scipy/MCMC)
                                         │
                               Visualization + Export
```

## CLI Reference

```
python -m src.pipeline [OPTIONS]

Modes:
  --mode single     Analyse a single target (default)
  --mode batch      Process multiple targets
  --mode train      Train the ML classifier

Options:
  --target NAME     Target name for single mode (default: "Pi Men")
  --sector N        TESS sector for batch mode
  --max-targets N   Limit number of targets in batch
  --max-sectors N   Max sectors to stitch per target (default: 5)
  --mcmc            Enable MCMC fitting for uncertainties
  --model-path PATH Path to trained classifier model
  --curated-csv PATH  CSV with labeled training data
  --toi-catalog     Use NASA TOI catalog for training
  --sde-threshold F Minimum SDE for detection (default: 7.0)
  --targets-file PATH  Text file with target names
```

## Output

| File | Description |
|------|-------------|
| `results/detection_results_latest.csv` | All detections with parameters & classification |
| `results/plots/*_diagnostic.png` | 7-panel diagnostic sheet per candidate |
| `results/plots/*_transits.png` | Individual transit gallery |
| `results/plots/*_corner.png` | MCMC posterior corner plots |
| `results/plots/sector_*_summary.png` | Sector-level summary |
| `results/pipeline_summary_report.txt` | Text summary report |

## Libraries Used

| Library | Purpose |
|---------|---------|
| lightkurve | TESS data download & manipulation |
| transitleastsquares | Transit detection (TLS) |
| batman-package | Physical transit model generation |
| emcee | MCMC posterior sampling |
| scikit-learn | Random Forest classifier |
| astropy | Time series, coordinates, FITS |
| astroquery | MAST/TIC catalog queries |
| scipy | Optimization, signal processing |
| matplotlib / seaborn | Visualization |
| corner | MCMC posterior corner plots |

## License

This project was developed for the ISRO Bharatiya Antariksh Hackathon 2026.
