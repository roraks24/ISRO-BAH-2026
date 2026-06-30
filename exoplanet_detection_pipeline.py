"""
Astroseekers — ISRO BAH 2026, Challenge 07
Exoplanet Transit Detection from TESS Light Curves

Quick single-target entry point.
For full pipeline (batch, ML training, MCMC), use:
    python -m src.pipeline --help

This script:
  1. Downloads a real light curve from MAST (via lightkurve)
  2. Cleans + detrends it (biweight filter, preserves transit dips)
  3. Runs Transit Least Squares (TLS) to detect the periodic dip
  4. Extracts features & classifies (ML or rule-based)
  5. Fits a batman transit model (with optional MCMC uncertainties)
  6. Generates a 7-panel diagnostic plot + transit gallery

Install once:
    pip install -r requirements.txt
"""

import sys
import os
import io
import logging
import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# Configure logging so the rich per-stage output is visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

from src.single_pipeline import analyse_single_target
from src.classifier import TransitClassifier
from src.report_generator import export_results_csv


def main():
    # ── Configuration ──────────────────────────────────────
    TARGET = "Pi Men"           # known exoplanet host (Pi Mensae c)
    MAX_SECTORS = 2             # sectors to stitch (use more for better SDE)
    DO_MCMC = False             # set True for MCMC uncertainties (slower)
    MODEL_PATH = "models/transit_classifier.joblib"

    # Load ML classifier if available
    classifier = None
    if os.path.exists(MODEL_PATH):
        classifier = TransitClassifier(model_path=MODEL_PATH)
        print("Loaded trained ML classifier.")
    else:
        print("No trained model found — using rule-based classifier.")
        print("Run `python -m src.pipeline --mode train --toi-catalog` to train.\n")

    # ── Run pipeline ───────────────────────────────────────
    result = analyse_single_target(
        TARGET,
        max_sectors=MAX_SECTORS,
        do_mcmc=DO_MCMC,
        classifier=classifier,
    )

    # Export results to CSV
    if result:
        df = pd.DataFrame(result)
        export_results_csv(df, "results")
        print("\nDone! Check results/ directory for plots and CSV output.")
    else:
        print("\nNo significant detections found.")


if __name__ == "__main__":
    main()