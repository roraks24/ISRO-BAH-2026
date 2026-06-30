"""
report_generator.py — Results export and summary report generation.

Exports:
  - CSV with all detected candidates and their parameters
  - Summary statistics report
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime


def export_results_csv(results_df, output_dir="results"):
    """
    Export detection results to a CSV file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Map to Challenge 07 categories
    from src.config import HACKATHON_CLASS_MAP
    if "classification" in results_df.columns:
        results_df["challenge_07_class"] = results_df["classification"].map(HACKATHON_CLASS_MAP).fillna("False Alarm")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"detection_results_{timestamp}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results exported to: {csv_path}")

    # Also save a "latest" copy
    latest_path = os.path.join(output_dir, "detection_results_latest.csv")
    results_df.to_csv(latest_path, index=False)

    return csv_path


def generate_summary_report(results_df, output_dir="results"):
    """
    Generate a text summary report of pipeline results.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "pipeline_summary_report.txt")

    lines = []
    lines.append("=" * 70)
    lines.append("  EXOPLANET TRANSIT DETECTION PIPELINE — SUMMARY REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("  ISRO BAH 2026 — Challenge 07 — Team Astroseekers")
    lines.append("=" * 70)
    lines.append("")

    n_total = len(results_df)
    lines.append(f"  Total targets processed:    {n_total}")
    lines.append("")

    # Classification breakdown
    if "classification" in results_df.columns:
        lines.append("  ┌─ Classification Summary (Detailed) ─────────────┐")
        class_counts = results_df["classification"].value_counts()
        for cls, count in class_counts.items():
            pct = count / n_total * 100
            lines.append(f"  │  {cls:<28s} {count:>5d} ({pct:>5.1f}%) │")
        lines.append("  └─────────────────────────────────────────────────┘")
        lines.append("")

    if "challenge_07_class" in results_df.columns:
        lines.append("  ┌─ Challenge 07 Official Classes ─────────────────┐")
        class_counts = results_df["challenge_07_class"].value_counts()
        for cls, count in class_counts.items():
            pct = count / n_total * 100
            lines.append(f"  │  {cls:<28s} {count:>5d} ({pct:>5.1f}%) │")
        lines.append("  └─────────────────────────────────────────────────┘")
        lines.append("")

    # Confidence breakdown
    if "confidence_level" in results_df.columns:
        lines.append("  ┌─ Confidence Level Distribution ──────────────────┐")
        conf_counts = results_df["confidence_level"].value_counts()
        for level in ["High", "Medium", "Low"]:
            count = conf_counts.get(level, 0)
            pct = count / n_total * 100 if n_total > 0 else 0
            lines.append(f"  │  {level:<28s} {count:>5d} ({pct:>5.1f}%) │")
        lines.append("  └─────────────────────────────────────────────────┘")
        lines.append("")

    # Planet candidates
    planets = results_df[results_df.get("classification", pd.Series()).isin(["planet_candidate", "massive_planet"])]
    if len(planets) > 0:
        lines.append(f"  ┌─ Planet Candidates ({len(planets)}) ─────────────────────────┐")
        lines.append(f"  │  {'Target':<18s} {'Period(d)':>10s} {'Depth(ppm)':>10s}"
                    f" {'SNR':>6s} {'Conf':>5s} │")
        lines.append(f"  │  {'─'*53} │")
        for _, row in planets.head(20).iterrows():
            name = str(row.get("target_name", row.get("tic_id", "?")))[:18]
            lines.append(
                f"  │  {name:<18s} {row.get('period_days', 0):>10.4f}"
                f" {row.get('depth_ppm', 0):>10.0f}"
                f" {row.get('snr', 0):>6.1f}"
                f" {row.get('confidence_score', 0):>5.0f} │"
            )
        if len(planets) > 20:
            lines.append(f"  │  ... and {len(planets) - 20} more                        │")
        lines.append("  └───────────────────────────────────────────────────────┘")
        lines.append("")

    # Parameter statistics
    if "period_days" in results_df.columns and len(results_df) > 0:
        lines.append("  ┌─ Parameter Statistics ──────────────────────────┐")
        for param, label, fmt in [
            ("period_days", "Period (days)", ".4f"),
            ("depth_ppm", "Depth (ppm)", ".0f"),
            ("duration_hr", "Duration (hours)", ".2f"),
            ("sde", "SDE", ".1f"),
            ("snr", "SNR", ".1f"),
        ]:
            if param in results_df.columns:
                vals = results_df[param].dropna()
                if len(vals) > 0:
                    lines.append(
                        f"  │  {label:<22s} "
                        f"min={vals.min():{fmt}}, "
                        f"med={vals.median():{fmt}}, "
                        f"max={vals.max():{fmt}} │"
                    )
        lines.append("  └─────────────────────────────────────────────────┘")
        lines.append("")

    # Methodology summary
    lines.append("  ┌─ Methodology ─────────────────────────────────────┐")
    lines.append("  │  Detection:      Transit Least Squares (TLS)      │")
    lines.append("  │  Detrending:     Biweight time-windowed filter    │")
    lines.append("  │  Classification: RF + rule-based hybrid           │")
    lines.append("  │  Model Fitting:  batman + scipy / emcee MCMC      │")
    lines.append("  │  Uncertainties:  MCMC posterior sampling           │")
    lines.append("  └───────────────────────────────────────────────────┘")

    lines.append("")
    lines.append("=" * 70)

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport saved to: {report_path}")
    return report_path
