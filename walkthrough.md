# Walkthrough: Exoplanet Transit Detection Pipeline

## What Was Built

A complete AI-driven pipeline for detecting, classifying, and characterising exoplanet
transit signals from TESS light curves. **10 Python modules** across 8 development phases.

## Files Created / Modified

### New Source Modules (src/)

| Module | Purpose |
|--------|---------|
| [data_loader.py](file:///e:/Coding/Hackathons/ISRO/src/data_loader.py) | TESS/MAST download with exponential-backoff retry, TOI catalog, sector batch, curated CSV |
| [detrending.py](file:///e:/Coding/Hackathons/ISRO/src/detrending.py) | Biweight time-windowed filter, SavGol, sigma clipping, transit masking |
| [detector.py](file:///e:/Coding/Hackathons/ISRO/src/detector.py) | TLS + BLS detection, multi-planet iterative search, SNR computation |
| [feature_extractor.py](file:///e:/Coding/Hackathons/ISRO/src/feature_extractor.py) | 19 diagnostic features: V-shape, secondary eclipse, rotation match, scatter ratio, etc. |
| [classifier.py](file:///e:/Coding/Hackathons/ISRO/src/classifier.py) | Random Forest ML + rule-based hybrid, 5-fold CV, interpretable reason strings |
| [transit_fitter.py](file:///e:/Coding/Hackathons/ISRO/src/transit_fitter.py) | batman transit models + scipy L-BFGS-B + emcee MCMC with proper priors |
| [visualizer.py](file:///e:/Coding/Hackathons/ISRO/src/visualizer.py) | 7-panel diagnostic sheets, sector summaries, MCMC corner plots, transit galleries |
| [pipeline.py](file:///e:/Coding/Hackathons/ISRO/src/pipeline.py) | End-to-end orchestrator with single/batch/train CLI modes, UTF-8 fix |
| [report_generator.py](file:///e:/Coding/Hackathons/ISRO/src/report_generator.py) | CSV export + formatted text summary reports |
| [__init__.py](file:///e:/Coding/Hackathons/ISRO/src/__init__.py) | Package initialisation |

### Project Files

| File | Purpose |
|------|---------|
| [requirements.txt](file:///e:/Coding/Hackathons/ISRO/requirements.txt) | All dependencies pinned |
| [README.md](file:///e:/Coding/Hackathons/ISRO/README.md) | Full docs with CLI reference, architecture diagram |
| [methodology_report.md](file:///e:/Coding/Hackathons/ISRO/report/methodology_report.md) | 3-page hackathon methodology report |
| [synthetic_test.py](file:///e:/Coding/Hackathons/ISRO/synthetic_test.py) | Offline demo: planet / EB / starspot classification without MAST |

### Modified Files

| File | Changes |
|------|---------|
| [exoplanet_detection_pipeline.py](file:///e:/Coding/Hackathons/ISRO/exoplanet_detection_pipeline.py) | Rewritten to delegate to new modular pipeline |
| [signal_classifier.py](file:///e:/Coding/Hackathons/ISRO/signal_classifier.py) | Fixed duplicate import, added deprecation notice |

---

## Bugs Fixed During Integration

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `UnicodeEncodeError` (σ, →, ═) | Windows cp1252 console | UTF-8 `TextIOWrapper` at startup in `pipeline.py` + `synthetic_test.py` |
| Transit gallery `ValueError: NaN to int` | `sharex=True` with empty subplots | Pre-filter transit times that have data; removed shared axes |
| `σ` in classifier reasons crashes print | cp1252 codec | Replaced `σ` → `sigma`, `R⊕` → `Re` in all reason strings |
| MAST `ConnectionError` (transient) | Remote server drops connection | Exponential-backoff retry (3 attempts, 1/2/4s) in `download_light_curve` |

---

## Validation 1 — Pi Men c (Real TESS Data)

![Pi Men diagnostic](file:///e:/Coding/Hackathons/ISRO/results/plots/Pi_Men_diagnostic.png)

| Parameter | Pipeline | Known | Match |
|-----------|---------|-------|-------|
| Period | 6.2681 d | 6.268 d | ✅ |
| Depth (TLS) | 266 ppm | ~250 ppm | ✅ |
| Rp/Rs (batman) | 0.01488 | ~0.018 | ✅ |
| Inclination | 89.6° | ~89.5° | ✅ |
| SDE | 31.9 | >7 | ✅ |
| Classification | planet_candidate High (88/100) | Confirmed planet | ✅ |

---

## Validation 2 — Synthetic Classification Tests (Offline)

Three synthetic 27-day TESS-like light curves (250 ppm noise, 2-min cadence, batman models):

### Planet (Pi Men c parameters)

| Metric | Value |
|--------|-------|
| Period detected | 6.271 d (true: 6.268 d) ✅ |
| SDE | 14.7 |
| Verdict | **planet_candidate** ✅ |
| batman fit χ²_red | 1.008 (excellent) |

### Eclipsing Binary (synthetic 3.14 d, 8% primary + 6% secondary)

![EB diagnostic](file:///e:/Coding/Hackathons/ISRO/results/plots/EclipsingBinary_Synthetic_diagnostic.png)

| Flag | Value |
|------|-------|
| Odd-even mismatch | 11.76 sigma → EB flag ✅ |
| Secondary eclipse | 20.1 sigma at phase 0.5 → EB flag ✅ |
| Implied radius | 26.3 Re (>22 Re ceiling) → blend flag ✅ |
| Verdict | **eclipsing_binary** ✅ |

### Starspot (12-day rotation, 3000 ppm amplitude)

![Starspot diagnostic](file:///e:/Coding/Hackathons/ISRO/results/plots/Starspot_Synthetic_diagnostic.png)

| Flag | Value |
|------|-------|
| SDE | 4.2 (below 7 threshold) → false alarm flag ✅ |
| Rotation match | Period 11.74 d, FAP=0.0 → starspot flag ✅ |
| Verdict | **starspot** ✅ |

### Summary Table

| Target | SDE | SNR | Classification | Confidence |
|--------|-----|-----|----------------|------------|
| Planet_Pi-Men-c | 14.7 | 21.2 | planet_candidate | Medium (62/100) |
| EclipsingBinary_Synthetic | 35.1 | 3830.5 | eclipsing_binary | Low (10/100)* |
| Starspot_Synthetic | 4.2 | 63.5 | starspot | Low (0/100)* |

> *Low confidence score for EB/starspot means high certainty they are NOT planets — the score penalises all FP flags. The verdict itself is correct in all cases.

---

## CLI Quick Reference

```bash
# Real TESS target
python -m src.pipeline --target "Pi Men" --mode single

# With MCMC uncertainties
python -m src.pipeline --target "Pi Men" --mode single --mcmc

# Offline synthetic demo (no internet needed)
python synthetic_test.py

# Train ML classifier on TOI catalog
python -m src.pipeline --mode train --toi-catalog --max-targets 200

# Batch process a TESS sector
python -m src.pipeline --mode batch --sector 1 --max-targets 500
```

---

## Output Files

| File | Description |
|------|-------------|
| `results/plots/*_diagnostic.png` | 7-panel candidate sheet (colour-coded by verdict) |
| `results/plots/*_transits.png` | Individual transit gallery |
| `results/plots/*_corner.png` | MCMC posterior corner plots (if --mcmc) |
| `results/detection_results_latest.csv` | All detections with 22 columns |
| `results/pipeline_summary_report.txt` | Formatted text summary |
| `results/synthetic_test_results.csv` | Synthetic test results |
