"""
config.py — Central configuration constants for the pipeline.

All tunable thresholds and directory paths in one place so that
single_pipeline, batch_pipeline, and training_pipeline stay in sync.
"""

# ── Detection thresholds ────────────────────────────────────────────
MIN_SDE = 7.0          # Signal Detection Efficiency floor (Hippke & Heller 2019)
MIN_SNR = 5.0          # Minimum transit SNR for a credible detection
MIN_TRANSITS = 2       # Minimum distinct transits for a reliable period

# ── Detrending ──────────────────────────────────────────────────────
DETREND_WINDOW = 0.5   # Biweight kernel half-width in days

# ── Classification ──────────────────────────────────────────────────
MAX_PLANET_RADIUS_REARTH = 22.0   # ~2 Jupiter radii — ceiling for "planet"
MASSIVE_PLANET_FLOOR_REARTH = 11.2  # ~1 Jupiter radius — floor for "massive planet"

# ── Directories ─────────────────────────────────────────────────────
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
MODELS_DIR = "models"
DATA_DIR = "data"

# ── Hackathon class labels ──────────────────────────────────────────
# Internal labels → official Challenge-07 categories
HACKATHON_CLASS_MAP = {
    "planet_candidate": "Planet",
    "massive_planet":   "Massive Planet",
    "eclipsing_binary": "Eclipsing Binary",
    "blend":            "False Alarm",
    "starspot":         "False Alarm",
    "false_alarm":      "False Alarm",
}
