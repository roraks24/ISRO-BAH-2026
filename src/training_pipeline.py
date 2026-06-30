import logging
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import (
    download_light_curve, download_sector_lightcurves,
    load_curated_dataset, load_toi_catalog, get_stellar_params,
)
from src.detrending import clean_lightcurve, detrend_lightcurve
from src.detector import run_tls, run_multi_planet_search
from src.feature_extractor import extract_features
from src.classifier import TransitClassifier, classify_rule_based
from src.transit_fitter import fit_transit_fast, fit_transit_mcmc
from src.visualizer import (
    plot_candidate_diagnostic, plot_sector_summary,
    plot_corner, plot_individual_transits,
)
from src.report_generator import export_results_csv, generate_summary_report

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR = "models"

logger = logging.getLogger(__name__)

from src.single_pipeline import ensure_dirs

def train_classifier_pipeline(curated_csv=None, use_toi=True,
                               max_samples=500):
    """
    Train the ML classifier on labeled data.
    """
    ensure_dirs()
    logger.info(f"\n{'='*60}")
    logger.info(f"  TRAINING ML CLASSIFIER")
    logger.info(f"{'='*60}\n")

    # Load labeled data
    if curated_csv is not None:
        df = load_curated_dataset(curated_csv)
    elif use_toi:
        df = load_toi_catalog()
        # Filter to usable labels (exclude ambiguous candidates)
        df = df[df["label"].isin(["planet", "false_positive", "false_alarm"])]
        # Map to our class labels
        label_map = {
            "planet": "planet_candidate",
            "false_positive": "eclipsing_binary",  # most FPs are EBs
            "false_alarm": "false_alarm",
        }
        df["label"] = df["label"].map(label_map)
    else:
        raise ValueError("Provide curated_csv or set use_toi=True")

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    logger.info(f"Training on {len(df)} labeled targets")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}\n")

    # Extract features for each target
    features_list = []
    labels = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        tic_id = str(row["tic_id"])
        try:
            lc, tid = download_light_curve(f"TIC {tic_id}", max_sectors=2)
            time_arr = np.asarray(lc.time.value, dtype=np.float64)
            flux_arr = np.asarray(lc.flux.value, dtype=np.float64)
            time_arr, flux_arr, _ = clean_lightcurve(time_arr, flux_arr)

            if len(time_arr) < 100:
                continue

            flat_flux, _ = detrend_lightcurve(time_arr, flux_arr)
            detection = run_tls(time_arr, flat_flux)

            stellar = get_stellar_params(tic_id)
            feats = extract_features(
                detection, time_arr, flat_flux,
                stellar_radius_rsun=stellar["radius_rsun"],
                stellar_teff=stellar["teff_k"],
                raw_flux=flux_arr,
            )

            features_list.append(feats)
            labels.append(row["label"])

        except Exception as e:
            continue

    if len(features_list) < 20:
        logger.info(f"Only {len(features_list)} usable samples — too few to train.")
        return None

    # Train
    clf = TransitClassifier()
    metrics = clf.train(features_list, labels)

    # Save model
    model_path = os.path.join(MODELS_DIR, "transit_classifier.joblib")
    clf.save_model(model_path)

    return clf, metrics

