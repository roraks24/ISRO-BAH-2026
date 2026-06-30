"""
multi_planet_demo.py — Demonstrates multi-planet detection on a synthetic
light curve containing TWO overlapping planet signals.

Shows the pipeline's iterative planet-masking capability:
  - Planet b: P=5.5 d, Rp/Rs=0.025 (Neptune-size)
  - Planet c: P=9.1 d, Rp/Rs=0.018 (sub-Neptune)

Usage:
    python multi_planet_demo.py
"""

import sys, os, io
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# UTF-8 fix for Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import batman
    HAS_BATMAN = True
except ImportError:
    HAS_BATMAN = False

from src.detector import run_tls, run_multi_planet_search
from src.feature_extractor import extract_features
from src.classifier import classify_rule_based
from src.transit_fitter import fit_transit_fast

RNG = np.random.default_rng(7)
CADENCE = 2 / (24 * 60)       # 2-minute cadence in days
DURATION = 54.0                # 2 TESS sectors
NOISE = 200e-6                 # 200 ppm

os.makedirs("results/plots", exist_ok=True)


def make_transit(time, period, t0, rp, a, inc):
    if not HAS_BATMAN:
        flux = np.ones_like(time)
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
        depth = rp**2
        dur = 2 * np.arcsin(1 / a) / (2 * np.pi / period) / period
        flux -= depth * (np.abs(phase) < dur / 2).astype(float)
        return flux
    params = batman.TransitParams()
    params.t0 = t0; params.per = period; params.rp = rp
    params.a = a; params.inc = inc; params.ecc = 0; params.w = 90
    params.u = [0.4, 0.25]; params.limb_dark = "quadratic"
    m = batman.TransitModel(params, time)
    return m.light_curve(params)


def main():
    print("\n" + "="*60)
    print("  MULTI-PLANET DETECTION DEMO")
    print("="*60)

    # Build synthetic 2-planet system
    time = np.arange(0, DURATION, CADENCE)

    print(f"\nGenerating synthetic 2-planet light curve...")
    print(f"  Planet b: P=5.5 d, Rp/Rs=0.025 (depth ~625 ppm)")
    print(f"  Planet c: P=9.1 d, Rp/Rs=0.018 (depth ~324 ppm)")
    print(f"  Duration: {DURATION:.0f} days | N={len(time):,} | Noise: {NOISE*1e6:.0f} ppm")

    flux_b = make_transit(time, period=5.5, t0=1.2, rp=0.025, a=12.0, inc=89.2)
    flux_c = make_transit(time, period=9.1, t0=3.7, rp=0.018, a=18.0, inc=88.7)
    flux = (flux_b - 1.0) + (flux_c - 1.0) + 1.0 + RNG.normal(0, NOISE, len(time))
    flux_err = np.full_like(flux, NOISE)

    # Run multi-planet search
    print("\nRunning iterative multi-planet TLS search...")
    detections = run_multi_planet_search(time, flux, max_planets=3, sde_threshold=6.0)

    print(f"\nFound {len(detections)} planet signal(s):")
    results = []
    for i, det in enumerate(detections):
        features = extract_features(det, time, flux, stellar_radius_rsun=1.0)
        classification = classify_rule_based(features)
        fit = None
        try:
            fit = fit_transit_fast(time, flux, flux_err, detection=det)
        except Exception:
            pass

        results.append((det, classification, fit))
        print(f"\n  Planet candidate {i+1}:")
        print(f"    Period:    {det.period:.4f} d")
        print(f"    Depth:     {det.depth_ppm:.0f} ppm")
        print(f"    Duration:  {det.duration*24:.2f} hr")
        print(f"    SDE:       {det.sde:.1f}  |  SNR: {det.snr:.1f}")
        print(f"    Class:     {classification.verdict} ({classification.confidence_level})")
        if fit:
            print(f"    Rp/Rs:     {fit.rp_rs:.5f}  (true: {[0.025, 0.018][min(i,1)]:.5f})")
            print(f"    chi2_red:  {fit.chi2_red:.3f}")

    # Plot all candidates on one figure
    _plot_multi_planet(time, flux, detections, results)
    print("\nPlot saved to results/plots/multi_planet_demo.png")

    # Match to injected periods
    print("\n" + "="*60)
    print("  RECOVERY CHECK")
    print("="*60)
    true_periods = {5.5: "b", 9.1: "c"}
    for det, cls, fit in results:
        for tp, name in true_periods.items():
            if abs(det.period - tp) / tp < 0.02 or abs(det.period - tp/2) / tp < 0.02:
                recovered = abs(det.period - tp) / tp < 0.02
                print(f"  Planet {name} (P={tp}d): {'RECOVERED' if recovered else 'HALF-PERIOD'} "
                      f"-> detected P={det.period:.4f}d [{cls.verdict}]")
                break
    print("="*60)


def _plot_multi_planet(time, flux, detections, results):
    n = len(detections)
    if n == 0:
        return

    fig = plt.figure(figsize=(16, 4 + 4 * n))
    gs = gridspec.GridSpec(1 + n, 2, figure=fig, hspace=0.45, wspace=0.3)

    colors = ["#e74c3c", "#3498db", "#27ae60", "#f39c12"]

    # Top panel: full light curve with all transit times marked
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(time, flux, '.', color='#888888', alpha=0.2, markersize=1, rasterized=True)
    for i, (det, cls, _) in enumerate(results):
        for tt in det.transit_times[:30]:
            ax0.axvline(tt, color=colors[i % len(colors)], alpha=0.4,
                       linewidth=0.8, label=f"Planet {i+1}" if tt == det.transit_times[0] else "")
    ax0.set_xlabel("Time (BJD)")
    ax0.set_ylabel("Relative Flux")
    ax0.set_title("Synthetic 2-Planet Light Curve — Transit Times Marked")
    handles = [plt.Line2D([0], [0], color=colors[i % len(colors)], linewidth=2,
                          label=f"Planet {i+1}: P={det.period:.3f}d")
               for i, (det, _, __) in enumerate(results)]
    ax0.legend(handles=handles, fontsize=9, loc="upper right")

    # Phase-folded plot for each planet
    for i, (det, cls, fit) in enumerate(results):
        ax = fig.add_subplot(gs[i + 1, 0])
        phase = ((time - det.t0) % det.period) / det.period
        phase[phase > 0.5] -= 1.0
        ax.plot(phase, flux, '.', color='#bdc3c7', alpha=0.2, markersize=1, rasterized=True)

        # Binned
        sort = np.argsort(phase)
        ps, fs = phase[sort], flux[sort]
        bins = np.linspace(-0.5, 0.5, 80)
        bcenters, bflux = [], []
        for j in range(len(bins)-1):
            m = (ps >= bins[j]) & (ps < bins[j+1])
            if m.sum() > 3:
                bcenters.append((bins[j]+bins[j+1])/2)
                bflux.append(np.median(fs[m]))
        if bcenters:
            ax.plot(bcenters, bflux, 'o', color=colors[i % len(colors)],
                   markersize=4, label="Binned")

        ax.set_xlabel("Phase")
        ax.set_ylabel("Relative Flux")
        ax.set_title(f"Planet {i+1}: P={det.period:.4f}d, Depth={det.depth_ppm:.0f}ppm "
                     f"| {cls.verdict}")
        ax.set_xlim(-0.15, 0.15)

        # Stats panel
        ax_txt = fig.add_subplot(gs[i + 1, 1])
        ax_txt.axis("off")
        txt = (f"SDE:     {det.sde:.1f}\n"
               f"SNR:     {det.snr:.1f}\n"
               f"N_trans: {det.n_transits}\n"
               f"Verdict: {cls.verdict}\n"
               f"Conf:    {cls.confidence_level} ({cls.confidence_score:.0f}/100)\n")
        if fit:
            txt += (f"\nFitted Rp/Rs: {fit.rp_rs:.5f}\n"
                    f"Fitted a/Rs:  {fit.a_rs:.2f}\n"
                    f"chi2_red:     {fit.chi2_red:.3f}")
        ax_txt.text(0.05, 0.9, txt, transform=ax_txt.transAxes,
                   fontfamily="monospace", fontsize=10, verticalalignment="top")

    fig.suptitle("Multi-Planet Detection Demo — Iterative TLS Masking",
                fontsize=14, fontweight="bold", y=0.98)

    fig.savefig("results/plots/multi_planet_demo.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
