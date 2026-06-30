"""
classifier.py — ML + rule-based hybrid transit signal classifier.

Classification targets:
    planet_candidate    — likely exoplanet transit
    eclipsing_binary    — two stars eclipsing each other
    blend               — diluted background EB or stellar companion
    starspot            — stellar rotation / spot modulation
    false_alarm         — instrumental artifact or noise

Two modes:
    1. ML mode (Random Forest) — trained on TOI catalog or curated dataset
    2. Rule-based fallback — enhanced version of the original signal_classifier.py
"""

import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score
    )
    from sklearn.preprocessing import LabelEncoder
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    class LabelEncoder:
        pass


# ── Import feature utilities ──
from src.feature_extractor import get_feature_names, features_to_array


CLASS_LABELS = [
    "planet_candidate", "massive_planet", "eclipsing_binary", "blend",
    "starspot", "false_alarm",
]

MAX_PLANET_RADIUS_REARTH = 22.0


@dataclass
class ClassificationResult:
    verdict: str               # predicted class
    confidence_level: str      # 'High', 'Medium', 'Low'
    confidence_score: float    # 0-100
    class_probabilities: dict  # {class_name: probability}
    reasons: List[str]         # interpretable explanations
    method: str                # 'ml' or 'rule_based'


# ═══════════════════════════════════════════════════════════════════════
#  ML CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════

class TransitClassifier:
    """
    Random Forest classifier for transit signal classification.
    Falls back to rule-based classification if no model is trained.
    """

    def __init__(self, model_path=None):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = get_feature_names()
        self.is_trained = False

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(self, features_list, labels, test_size=0.2):
        """
        Train the Random Forest classifier.

        Parameters
        ----------
        features_list : list of dict
            Each dict is a feature vector from extract_features().
        labels : list of str
            Ground truth labels.

        Returns
        -------
        metrics : dict
            Training metrics including accuracy, classification report.
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for ML classification.")

        # Convert to arrays
        X = np.array([features_to_array(f, self.feature_names)
                       for f in features_list])
        # Replace NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Encode labels
        y = self.label_encoder.fit_transform(labels)

        # Train with cross-validation
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        # Cross-validated predictions for evaluation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(self.model, X, y, cv=cv)

        # Train on full dataset
        self.model.fit(X, y)
        self.is_trained = True

        # Metrics
        target_names = self.label_encoder.classes_
        report = classification_report(
            y, y_pred_cv, target_names=target_names, output_dict=True
        )
        cm = confusion_matrix(y, y_pred_cv)
        accuracy = accuracy_score(y, y_pred_cv)

        # Feature importances
        importances = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))

        metrics = {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "feature_importances": importances,
            "n_samples": len(y),
            "class_distribution": dict(zip(
                target_names,
                [int((y == i).sum()) for i in range(len(target_names))]
            )),
        }

        print(f"\n{'='*60}")
        print(f"ML Classifier Training Results")
        print(f"{'='*60}")
        print(f"Accuracy (5-fold CV): {accuracy:.3f}")
        print(f"Samples: {len(y)}")
        print(classification_report(y, y_pred_cv, target_names=target_names))
        print(f"\nTop 5 features by importance:")
        sorted_imp = sorted(importances.items(), key=lambda x: -x[1])
        for name, imp in sorted_imp[:5]:
            print(f"  {name}: {imp:.4f}")

        return metrics

    def predict(self, features):
        """
        Predict the class of a transit signal.

        Parameters
        ----------
        features : dict
            Feature vector from extract_features().

        Returns
        -------
        ClassificationResult
        """
        if not self.is_trained:
            return classify_rule_based(features)

        X = features_to_array(features, self.feature_names).reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        proba = self.model.predict_proba(X)[0]
        class_names = self.label_encoder.classes_
        probabilities = dict(zip(class_names, proba.tolist()))

        # ML verdict
        ml_class_idx = np.argmax(proba)
        ml_verdict = class_names[ml_class_idx]
        ml_confidence = proba[ml_class_idx]

        # Rule-based override for extreme cases
        rule_result = classify_rule_based(features)
        reasons = []

        final_verdict = ml_verdict
        # Override ML if rule-based has very strong evidence
        if (rule_result.confidence_score < 20 and
                rule_result.verdict not in ["planet_candidate", "massive_planet"] and
                ml_verdict in ["planet_candidate", "massive_planet"]):
            final_verdict = rule_result.verdict
            reasons.append(
                f"ML predicted planet ({ml_confidence:.0%}) but rule-based "
                f"override to {rule_result.verdict} due to strong FP evidence"
            )
            reasons.extend(rule_result.reasons)
        else:
            reasons.append(f"ML prediction: {ml_verdict} ({ml_confidence:.0%})")
            if rule_result.verdict != ml_verdict and rule_result.verdict not in ["planet_candidate", "massive_planet"]:
                reasons.append(
                    f"Rule-based check flagged: {rule_result.verdict} "
                    f"({', '.join(rule_result.reasons)})"
                )

        # Confidence level
        if ml_confidence >= 0.80:
            confidence_level = "High"
        elif ml_confidence >= 0.50:
            confidence_level = "Medium"
        else:
            confidence_level = "Low"

        return ClassificationResult(
            verdict=final_verdict,
            confidence_level=confidence_level,
            confidence_score=float(ml_confidence * 100),
            class_probabilities=probabilities,
            reasons=reasons,
            method="ml+rules",
        )

    def save_model(self, path):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump({
            "model": self.model,
            "label_encoder": self.label_encoder,
            "feature_names": self.feature_names,
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load a trained model from disk."""
        data = joblib.load(path)
        self.model = data["model"]
        self.label_encoder = data["label_encoder"]
        self.feature_names = data["feature_names"]
        self.is_trained = True
        print(f"Model loaded from {path}")


# ═══════════════════════════════════════════════════════════════════════
#  RULE-BASED CLASSIFIER (enhanced fallback)
# ═══════════════════════════════════════════════════════════════════════

def classify_rule_based(features):
    """
    Enhanced rule-based classifier. Evaluates all diagnostic checks
    and combines them into a verdict + confidence level.

    This is the interpretable fallback that works with zero training data.
    """
    flags = []  # (category, reason_text, penalty)

    # 1. Odd-even mismatch → eclipsing binary
    oe = features.get("odd_even_mismatch", 0)
    if oe > 3:
        penalty = min(oe * 5, 40)
        flags.append((
            "eclipsing_binary",
            f"odd-even depth mismatch = {oe:.2f} sigma (>3 sigma threshold)",
            penalty,
        ))

    # 2. Secondary eclipse → eclipsing binary
    sec_sigma = features.get("secondary_eclipse_sigma", 0)
    if sec_sigma > 5:
        penalty = min(sec_sigma * 3, 40)
        flags.append((
            "eclipsing_binary",
            f"secondary eclipse detected at {sec_sigma:.1f} sigma near phase 0.5",
            penalty,
        ))

    # 3. V-shaped transit → eclipsing binary
    v_shape = features.get("v_shape_metric", 0.5)
    if v_shape > 0.85:
        flags.append((
            "eclipsing_binary",
            f"V-shaped transit (v_shape={v_shape:.2f}, >0.85 threshold)",
            25,
        ))

    # 4. Implied radius too large → blend / stellar companion
    rp_earth = features.get("rp_earth_radii", 0)
    if rp_earth > MAX_PLANET_RADIUS_REARTH:
        excess = rp_earth - MAX_PLANET_RADIUS_REARTH
        penalty = min(excess * 0.5, 40)
        flags.append((
            "blend",
            f"implied radius {rp_earth:.1f} Re exceeds {MAX_PLANET_RADIUS_REARTH} Re ceiling",
            penalty,
        ))

    # 5. Very deep transit (>5%) → likely EB or blend
    depth_ppm = features.get("depth_ppm", 0)
    if depth_ppm > 50000:  # 5%
        flags.append((
            "eclipsing_binary",
            f"transit depth {depth_ppm:.0f} ppm (>5%) — likely stellar eclipse",
            30,
        ))

    # 6. Rotation period matches transit period OR very high LS power for a marginal signal → starspot
    rot_match = features.get("rotation_near_transit_period", 0)
    rot_period = features.get("rotation_period_days", np.nan)
    rot_fap = features.get("rotation_peak_fap", np.nan)
    rot_power = features.get("rotation_peak_power", np.nan)
    sde = features.get("sde", 0)
    if rot_match > 0.5 or (np.isfinite(rot_power) and rot_power > 0.5 and rot_fap < 0.01 and sde < 10.0):
        if rot_match > 0.5:
            reason = f"dip period matches stellar rotation ({rot_period:.3f}d, FAP={rot_fap:.2e})"
        else:
            reason = f"strong sinusoidal stellar activity (LS power={rot_power:.2f}, FAP={rot_fap:.2e})"
        flags.append(("starspot", reason, 30))

    # 7. Weak detection → low confidence
    sde = features.get("sde", 0)
    if sde < 7:
        flags.append((
            "false_alarm",
            f"SDE = {sde:.1f} below significance threshold (7)",
            20,
        ))

    # 8. Very few transits → low confidence
    n_transits = features.get("n_transits", 0)
    if n_transits < 2:
        flags.append((
            "false_alarm",
            f"only {n_transits} transit(s) observed — single-event unreliable",
            15,
        ))

    # ── Hot-Jupiter rescue ──────────────────────────────────────────
    # Ultra-hot Jupiters (P < 2d) show genuine secondary thermal eclipses
    # and rotation aliases. If odd-even is low (<3σ), rescue to planet_candidate.
    period = features.get("period_days", np.nan)
    rp_earth = features.get("rp_earth_radii", 0)
    if (np.isfinite(period) and period < 2.0
            and oe < 3.0
            and rp_earth < 200.0):        # exclude real EBs with stellar-size objects
        # Check secondary eclipse depth relative to primary transit depth.
        # Genuine EBs have secondary ≈ primary; hot Jupiters have tiny thermal emission.
        sec_depth = features.get('secondary_eclipse_depth', 0.0)
        primary_depth = features.get('depth_ppm', 1.0) * 1e-6  # convert ppm to fractional
        sec_to_primary_ratio = sec_depth / primary_depth if primary_depth > 0 else 0.0
        if sec_to_primary_ratio < 0.05:
            # Downgrade EB / starspot flags caused by secondary eclipse or rotation alias
            flags = [f for f in flags
                     if not ((f[0] == "eclipsing_binary" and "secondary eclipse" in f[1]) or
                             (f[0] == "starspot" and "rotation" in f[1]))]
            if flags and all(f[0] == "eclipsing_binary" for f in flags):
                # Only remaining EB flags — check if they're depth-based
                flags = [f for f in flags if "depth" in f[1]]

    # ── Starspot rescue ──────────────────────────────────────────
    # Sinusoidal stellar rotation signals produce a large *apparent*
    # odd-even mismatch because TLS samples different phases of the
    # sinusoid as "odd" and "even" transits. This is an artifact of
    # the sinusoidal shape, not a real EB signature. When rotation
    # clearly matches AND the signal looks sinusoidal (long duration
    # relative to period — real transits have dur/period < 0.04,
    # sinusoidal signals typically have dur/period > 0.05), downgrade
    # the spurious odd-even EB flag.
    dur_hr = features.get("duration_hr", 0)
    period = features.get("period_days", np.nan)
    if (rot_match > 0.5
            and np.isfinite(period) and period > 0
            and dur_hr / (period * 24) > 0.05):
        # Remove odd-even EB flags — they are artifacts of sinusoidal shape
        flags = [f for f in flags if not (
            f[0] == "eclipsing_binary" and "odd-even" in f[1]
        )]

    # ── Verdict: priority order ──
    categories = [f[0] for f in flags]
    if "eclipsing_binary" in categories:
        verdict = "eclipsing_binary"
    elif "blend" in categories:
        verdict = "blend"
    elif "starspot" in categories:
        verdict = "starspot"
    elif "false_alarm" in categories:
        verdict = "false_alarm"
    else:
        rp_earth = features.get("rp_earth_radii", 0)
        if rp_earth >= 11.2:
            verdict = "massive_planet"
        else:
            verdict = "planet_candidate"


    # ── Confidence score ──
    base_score = float(np.clip((sde - 7) / (40 - 7) * 50 + 50, 0, 100))
    total_penalty = sum(f[2] for f in flags)
    confidence_score = float(np.clip(base_score - total_penalty, 0, 100))

    if confidence_score >= 70:
        confidence_level = "High"
    elif confidence_score >= 40:
        confidence_level = "Medium"
    else:
        confidence_level = "Low"

    if flags:
        reasons = [f[1] for f in flags]
    else:
        if verdict == "massive_planet":
            reasons = [f"passed all checks, but has a large radius of {features.get('rp_earth_radii', 0):.2f} Re (>=11.2 Re floor for massive planets)"]
        else:
            reasons = [
                "passed all diagnostic checks (odd-even, secondary eclipse, "
                "V-shape, radius, rotation)"
            ]

    # Dummy probabilities for interface compatibility
    probabilities = {c: 0.0 for c in CLASS_LABELS}
    probabilities[verdict] = confidence_score / 100.0
    if verdict not in ["planet_candidate", "massive_planet"]:
        probabilities["planet_candidate"] = (100 - confidence_score) / 100.0

    return ClassificationResult(
        verdict=verdict,
        confidence_level=confidence_level,
        confidence_score=confidence_score,
        class_probabilities=probabilities,
        reasons=reasons,
        method="rule_based",
    )
