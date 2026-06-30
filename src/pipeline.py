"""
pipeline.py — End-to-end orchestrator for the exoplanet detection pipeline.
Challenge 07 — ISRO BAH 2026 — Team Astroseekers

Modes:
  - single:  Analyse a single target (e.g. "Pi Men")
  - batch:   Process multiple targets from a sector or curated dataset
  - train:   Train the ML classifier on TOI catalog / curated labels

Usage:
    python -m src.pipeline --target "Pi Men" --mode single
    python -m src.pipeline --sector 1 --mode batch --max-targets 500
    python -m src.pipeline --mode train --toi-catalog
"""

import os
import sys
import io
import argparse
import pandas as pd

# Fix Windows console encoding (cp1252 can't print Unicode box-drawing chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )
    except Exception:
        pass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.single_pipeline import ensure_dirs, analyse_single_target
from src.batch_pipeline import analyse_batch
from src.training_pipeline import train_classifier_pipeline
from src.classifier import TransitClassifier
from src.report_generator import export_results_csv

RESULTS_DIR = "results"


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Exoplanet Transit Detection Pipeline — ISRO BAH 2026"
    )
    parser.add_argument("--mode", choices=["single", "batch", "train"],
                        default="single", help="Pipeline mode")
    parser.add_argument("--target", type=str, default="Pi Men",
                        help="Target name for single mode")
    parser.add_argument("--sector", type=int, default=None,
                        help="TESS sector number for batch mode")
    parser.add_argument("--max-targets", type=int, default=None,
                        help="Max targets in batch mode")
    parser.add_argument("--max-sectors", type=int, default=5,
                        help="Max sectors to stitch in single mode")
    parser.add_argument("--mcmc", action="store_true",
                        help="Run MCMC fitting (slower but gives uncertainties)")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained classifier model")
    parser.add_argument("--curated-csv", type=str, default=None,
                        help="Path to curated labeled dataset CSV")
    parser.add_argument("--toi-catalog", action="store_true",
                        help="Use TOI catalog for training")
    parser.add_argument("--sde-threshold", type=float, default=7.0,
                        help="Minimum SDE for a significant detection")
    parser.add_argument("--targets-file", type=str, default=None,
                        help="Text file with target names (one per line)")

    args = parser.parse_args()

    # Load classifier if available
    classifier = None
    if args.model_path and os.path.exists(args.model_path):
        classifier = TransitClassifier(model_path=args.model_path)

    ensure_dirs()

    if args.mode == "single":
        result = analyse_single_target(
            args.target, max_sectors=args.max_sectors,
            do_mcmc=args.mcmc, classifier=classifier,
        )
        # Save single result
        if result:
            df = pd.DataFrame(result)
            export_results_csv(df, RESULTS_DIR)
        else:
            print("\nNo significant detections found.")

    elif args.mode == "batch":
        if args.targets_file:
            with open(args.targets_file) as f:
                targets = [line.strip() for line in f if line.strip()]
            analyse_batch(target_list=targets, classifier=classifier)
        elif args.sector:
            analyse_batch(sector=args.sector, max_targets=args.max_targets,
                         classifier=classifier,
                         sde_threshold=args.sde_threshold)
        else:
            print("Batch mode requires --sector or --targets-file")

    elif args.mode == "train":
        train_classifier_pipeline(
            curated_csv=args.curated_csv,
            use_toi=args.toi_catalog,
            max_samples=args.max_targets or 500,
        )


if __name__ == "__main__":
    main()
