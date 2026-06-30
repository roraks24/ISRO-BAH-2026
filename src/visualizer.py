"""
visualizer.py — Publication-quality multi-panel diagnostic plots.

Generates:
  - 7-panel candidate diagnostic sheet
  - Sector-level summary plots
  - MCMC corner plots
  - Individual transit gallery
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import os


# ── Style defaults ──
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})


def plot_candidate_diagnostic(time, flux, detection, fit_result,
                              classification, features, target_name,
                              save_dir="results/plots", dqi=None):
    """
    Generate the full 7-panel diagnostic sheet for a transit candidate.

    Layout (3 rows × 3 columns, last column spans):
      [1] Raw/detrended light curve  [2] TLS periodogram   [7] Summary box
      [3] Phase-folded + model       [4] Odd-even compare   ...
      [5] Residuals                  [6] Secondary eclipse   ...
    """
    os.makedirs(save_dir, exist_ok=True)
    results = detection.raw_results

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, width_ratios=[1, 1, 0.8],
                           hspace=0.35, wspace=0.3)

    # ── Panel 1: Detrended light curve with transits marked ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, flux, '.', color='#888888', alpha=0.3, markersize=1,
             rasterized=True)
    # Mark transit times
    for tt in detection.transit_times[:20]:
        ax1.axvline(tt, color='#e74c3c', alpha=0.4, linewidth=0.8)
    ax1.set_xlabel("Time (BJD)")
    ax1.set_ylabel("Relative Flux")
    ax1.set_title("Detrended Light Curve")

    # ── Panel 2: TLS/BLS periodogram ──
    ax2 = fig.add_subplot(gs[0, 1])
    if results is not None and hasattr(results, "periods"):
        ax2.plot(results.periods, results.power, color='#2c3e50', linewidth=0.8)
        ax2.axvline(detection.period, color='#e74c3c', linewidth=1.5,
                    label=f"P={detection.period:.4f}d")
        ax2.axhline(7.0, color='#e67e22', linestyle='--', alpha=0.7,
                    label="SDE=7 threshold")
        ax2.set_xlabel("Period (days)")
        ax2.set_ylabel("SDE")
        ax2.set_title("TLS Periodogram")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "Periodogram\nnot available", transform=ax2.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
        ax2.set_title("TLS Periodogram")

    # ── Panel 3: Phase-folded transit + model fit ──
    ax3 = fig.add_subplot(gs[1, 0])
    if results is not None:
        phase = np.asarray(results.folded_phase)
        folded_y = np.asarray(results.folded_y)
        ax3.plot(phase, folded_y, '.', color='#bdc3c7', alpha=0.3,
                 markersize=2, rasterized=True, label="Data")

        # Binned data
        from src.transit_fitter import bin_phase_curve, fold_lightcurve
        try:
            sort = np.argsort(phase)
            bp, bf, be = bin_phase_curve(phase[sort], folded_y[sort], n_bins=150)
            ax3.errorbar(bp, bf, yerr=be, fmt='o', color='#2980b9',
                        markersize=3, alpha=0.8, label="Binned", capsize=0)
        except Exception:
            pass

        # Model overlay
        if hasattr(results, "model_folded_phase"):
            ax3.plot(results.model_folded_phase, results.model_folded_model,
                    color='#e74c3c', linewidth=2, label="Best-fit model", zorder=5)

        ax3.set_xlabel("Phase")
        ax3.set_ylabel("Relative Flux")
        ax3.set_title("Phase-Folded Transit")
        ax3.legend(fontsize=8, loc="lower right")

    # ── Panel 4: Odd vs Even transits ──
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_odd_even(ax4, time, flux, detection)

    # ── Panel 5: Residuals ──
    ax5 = fig.add_subplot(gs[2, 0])
    if results is not None:
        phase = np.asarray(results.folded_phase)
        folded_y = np.asarray(results.folded_y)
        if hasattr(results, "model_folded_phase") and hasattr(results, "model_folded_model"):
            from scipy.interpolate import interp1d
            try:
                model_interp = interp1d(
                    results.model_folded_phase, results.model_folded_model,
                    bounds_error=False, fill_value=1.0
                )
                model_at_data = model_interp(phase)
                residuals = folded_y - model_at_data
                ax5.plot(phase, residuals * 1e6, '.', color='#27ae60',
                        alpha=0.3, markersize=1, rasterized=True)
                ax5.axhline(0, color='#e74c3c', linewidth=1)
                ax5.set_ylabel("Residuals (ppm)")
            except Exception:
                ax5.text(0.5, 0.5, "Residuals\nnot available",
                        transform=ax5.transAxes, ha='center', va='center')
        else:
            ax5.text(0.5, 0.5, "Model data\nnot available",
                    transform=ax5.transAxes, ha='center', va='center')
    ax5.set_xlabel("Phase")
    ax5.set_title("Fit Residuals")

    # ── Panel 6: Secondary eclipse check ──
    ax6 = fig.add_subplot(gs[2, 1])
    _plot_secondary_eclipse(ax6, results)

    # ── Panel 7: Classification summary (spanning right column) ──
    ax7 = fig.add_subplot(gs[:, 2])
    _plot_summary_box(ax7, target_name, detection, fit_result,
                      classification, features, dqi=dqi)

    # ── Super title ──
    fig.suptitle(
        f"{target_name} — Transit Candidate Diagnostic",
        fontsize=16, fontweight="bold", y=0.98
    )

    save_path = os.path.join(save_dir, f"{target_name.replace(' ', '_')}_diagnostic.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved diagnostic plot: {save_path}")
    plt.close(fig)
    return save_path


def _plot_odd_even(ax, time, flux, detection):
    """Plot odd and even transits overlaid for EB diagnostic."""
    period = detection.period
    t0 = detection.t0
    dur = detection.duration

    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    # Determine odd/even based on transit number
    transit_num = np.round((time - t0) / period).astype(int)
    is_odd = transit_num % 2 == 1
    is_even = transit_num % 2 == 0

    # Near transit
    near_transit = np.abs(phase) < 0.1

    odd_mask = is_odd & near_transit
    even_mask = is_even & near_transit

    if odd_mask.sum() > 5:
        ax.plot(phase[odd_mask], flux[odd_mask], '.', color='#e74c3c',
                alpha=0.4, markersize=2, label="Odd transits")
    if even_mask.sum() > 5:
        ax.plot(phase[even_mask], flux[even_mask], '.', color='#3498db',
                alpha=0.4, markersize=2, label="Even transits")

    ax.set_xlabel("Phase")
    ax.set_ylabel("Relative Flux")
    ax.set_title(f"Odd vs Even (Δ={detection.odd_even_mismatch:.1f}σ)")
    ax.legend(fontsize=8)


def _plot_secondary_eclipse(ax, results):
    """Zoom on phase 0.4-0.6 to check for secondary eclipse."""
    if results is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha='center', va='center')
        return

    phase = np.asarray(results.folded_phase)
    flux = np.asarray(results.folded_y)

    mask = (phase > 0.3) & (phase < 0.7)
    if mask.sum() < 10:
        ax.text(0.5, 0.5, "Insufficient data\nnear phase 0.5",
                transform=ax.transAxes, ha='center', va='center')
        return

    ax.plot(phase[mask], flux[mask], '.', color='#95a5a6', alpha=0.3,
            markersize=2, rasterized=True)

    # Bin the data around phase 0.5
    from src.transit_fitter import bin_phase_curve
    try:
        sort = np.argsort(phase[mask])
        bp, bf, be = bin_phase_curve(phase[mask][sort], flux[mask][sort],
                                      n_bins=50)
        ax.errorbar(bp, bf, yerr=be, fmt='o', color='#8e44ad',
                    markersize=3, capsize=0)
    except Exception:
        pass

    ax.axvline(0.5, color='#e74c3c', linestyle='--', alpha=0.5,
               label="Phase 0.5")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Relative Flux")
    ax.set_title("Secondary Eclipse Check")
    ax.legend(fontsize=8)


def _plot_summary_box(ax, target_name, detection, fit_result,
                      classification, features, dqi=None):
    """Text summary panel with classification verdict and parameters."""
    ax.axis("off")

    # Background colour based on verdict
    colors = {
        "planet_candidate": "#27ae60",
        "massive_planet": "#2980b9",
        "eclipsing_binary": "#e74c3c",
        "blend": "#e67e22",
        "starspot": "#f39c12",
        "false_alarm": "#95a5a6",
    }
    verdict_color = colors.get(classification.verdict, "#95a5a6")

    lines = []
    lines.append(f"{'═' * 30}")
    lines.append(f"  TARGET: {target_name}")
    lines.append(f"{'═' * 30}")
    lines.append(f"")
    lines.append(f"  ▸ VERDICT: {classification.verdict.upper()}")
    lines.append(f"  ▸ Confidence: {classification.confidence_level}")
    lines.append(f"    ({classification.confidence_score:.0f}/100)")
    lines.append(f"  ▸ Method: {classification.method}")
    if dqi is not None:
        lines.append(f"  ▸ DQI: {dqi:.1f}/100")
    lines.append(f"")
    lines.append(f"{'─' * 30}")
    lines.append(f"  DETECTION METRICS")
    lines.append(f"{'─' * 30}")
    lines.append(f"  SDE:       {detection.sde:.1f}")
    lines.append(f"  SNR:       {detection.snr:.1f}")
    lines.append(f"  N_transits: {detection.n_transits}")
    lines.append(f"")
    lines.append(f"{'─' * 30}")
    lines.append(f"  FITTED PARAMETERS")
    lines.append(f"{'─' * 30}")

    if fit_result is not None:
        p = fit_result
        if p.period_err > 0:
            lines.append(f"  Period:    {p.period:.6f} ± {p.period_err:.6f} d")
        else:
            lines.append(f"  Period:    {p.period:.6f} d")

        if p.rp_rs_err > 0:
            lines.append(f"  Rp/Rs:     {p.rp_rs:.5f} ± {p.rp_rs_err:.5f}")
        else:
            lines.append(f"  Rp/Rs:     {p.rp_rs:.5f}")

        lines.append(f"  Depth:     {p.depth_ppm:.0f} ppm")

        if p.duration_hr > 0:
            lines.append(f"  Duration:  {p.duration_hr:.2f} hr")

        lines.append(f"  a/Rs:      {p.a_rs:.2f}")
        lines.append(f"  inc:       {p.inclination:.1f}°")
        lines.append(f"  b:         {p.impact_parameter:.3f}")
        lines.append(f"  χ²_red:    {p.chi2_red:.3f}")
    else:
        lines.append(f"  Period:    {detection.period:.6f} d")
        lines.append(f"  Depth:     {detection.depth_ppm:.0f} ppm")
        lines.append(f"  Duration:  {detection.duration*24:.2f} hr")

    lines.append(f"")
    lines.append(f"{'─' * 30}")
    lines.append(f"  DIAGNOSTICS")
    lines.append(f"{'─' * 30}")
    lines.append(f"  Odd-Even:  {detection.odd_even_mismatch:.1f}σ")

    sec_sig = features.get("secondary_eclipse_sigma", 0)
    lines.append(f"  Sec. Ecl:  {sec_sig:.1f}σ")

    v_shape = features.get("v_shape_metric", 0)
    lines.append(f"  V-shape:   {v_shape:.2f}")

    rp = features.get("rp_earth_radii", 0)
    lines.append(f"  Rp:        {rp:.1f} Re")

    text = "\n".join(lines)

    # Draw coloured header rectangle
    rect = FancyBboxPatch(
        (0.05, 0.88), 0.9, 0.10,
        boxstyle="round,pad=0.02",
        facecolor=verdict_color, alpha=0.3,
        transform=ax.transAxes
    )
    ax.add_patch(rect)

    ax.text(0.05, 0.85, text, transform=ax.transAxes,
            fontfamily="monospace", fontsize=8.5,
            verticalalignment="top")


def plot_sector_summary(results_df, sector_num, save_dir="results/plots"):
    """
    Overview plots for an entire sector's detections.
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Period histogram
    ax = axes[0, 0]
    periods = results_df["period_days"].dropna()
    ax.hist(periods, bins=50, color='#3498db', edgecolor='white', alpha=0.8)
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Count")
    ax.set_title("Period Distribution")

    # Depth histogram
    ax = axes[0, 1]
    depths = results_df["depth_ppm"].dropna()
    ax.hist(depths, bins=50, color='#e74c3c', edgecolor='white', alpha=0.8)
    ax.set_xlabel("Depth (ppm)")
    ax.set_ylabel("Count")
    ax.set_title("Transit Depth Distribution")
    ax.set_xscale("log")

    # SDE distribution
    ax = axes[1, 0]
    sdes = results_df["sde"].dropna()
    ax.hist(sdes, bins=50, color='#27ae60', edgecolor='white', alpha=0.8)
    ax.axvline(7.0, color='#e74c3c', linewidth=2, linestyle='--',
               label="SDE=7 threshold")
    ax.set_xlabel("SDE")
    ax.set_ylabel("Count")
    ax.set_title("Signal Detection Efficiency")
    ax.legend()

    # Classification pie chart
    ax = axes[1, 1]
    if "classification" in results_df.columns:
        counts = results_df["classification"].value_counts()
        colors_map = {
            "planet_candidate": "#27ae60",
            "eclipsing_binary": "#e74c3c",
            "blend": "#e67e22",
            "starspot": "#f39c12",
            "false_alarm": "#95a5a6",
        }
        pie_colors = [colors_map.get(c, "#bdc3c7") for c in counts.index]
        ax.pie(counts.values, labels=counts.index, colors=pie_colors,
               autopct='%1.1f%%', startangle=90)
        ax.set_title("Classification Distribution")
    else:
        ax.text(0.5, 0.5, "No classification data", transform=ax.transAxes,
                ha='center', va='center')

    fig.suptitle(f"TESS Sector {sector_num} — Detection Summary",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"sector_{sector_num}_summary.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved sector summary: {save_path}")
    plt.close(fig)
    return save_path


def plot_corner(mcmc_samples, labels, target_name, save_dir="results/plots"):
    """
    Generate MCMC posterior corner plot.
    """
    os.makedirs(save_dir, exist_ok=True)
    try:
        import corner
        fig = corner.corner(
            mcmc_samples, labels=labels,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 10},
            color="#3498db",
        )
        fig.suptitle(f"{target_name} — MCMC Posteriors", fontsize=14, y=1.02)
        save_path = os.path.join(save_dir,
                                  f"{target_name.replace(' ', '_')}_corner.png")
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved corner plot: {save_path}")
        plt.close(fig)
        return save_path
    except ImportError:
        print("  corner package not installed — skipping corner plot.")
        return None


def plot_individual_transits(time, flux, detection, target_name,
                             max_transits=12, save_dir="results/plots"):
    """
    Plot each individual transit event in a gallery grid.
    """
    os.makedirs(save_dir, exist_ok=True)
    transit_times = detection.transit_times
    if len(transit_times) == 0:
        return None

    half_window = detection.duration * 3  # show 3x the transit duration
    if half_window <= 0:
        half_window = 0.1  # fallback: ~2.4 hours

    # Filter to only transit times that actually have data nearby
    valid_transits = []
    for tt in transit_times:
        mask = np.abs(time - tt) < half_window
        if mask.sum() >= 3:
            valid_transits.append(tt)
    
    if len(valid_transits) == 0:
        return None

    n_show = min(len(valid_transits), max_transits)
    ncols = min(4, n_show)
    nrows = int(np.ceil(n_show / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_show):
        ax = axes[i]
        tt = valid_transits[i]
        mask = np.abs(time - tt) < half_window

        t_local = (time[mask] - tt) * 24  # hours from mid-transit
        ax.plot(t_local, flux[mask], '.', color='#2c3e50', markersize=3)
        ax.axvline(0, color='#e74c3c', linewidth=0.8, alpha=0.5)
        ax.set_title(f"Transit {i+1}", fontsize=9)
        if i >= n_show - ncols:
            ax.set_xlabel("Hours from mid-transit")

    # Hide unused axes
    for i in range(n_show, len(axes)):
        axes[i].axis('off')

    fig.suptitle(f"{target_name} -- Individual Transits", fontsize=14)
    plt.tight_layout()

    save_path = os.path.join(save_dir,
                              f"{target_name.replace(' ', '_')}_transits.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved transit gallery: {save_path}")
    plt.close(fig)
    return save_path
