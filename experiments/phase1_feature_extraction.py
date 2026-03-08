"""
phase1_feature_extraction.py — Phase 1 validation experiment.

Goal
Verify that the spectral analysis pipeline is correctly implemented by
running two extreme synthetic stimuli through the AVR computation:

1. Constant image  → AVR must be < 0.05  (all energy at DC)
2. Checkerboard    → AVR must be > 0.80  (energy at Nyquist)

If either criterion fails, the experiment exits with a non-zero status
code and a clear error message — never a silent pass.

Three figures are saved to results/figures/:
  - constant_power_spectrum.png   : 2-D power spectrum of the constant image
  - checkerboard_power_spectrum.png : 2-D power spectrum of the checkerboard
  - avr_comparison.png            : Bar chart comparing the two AVR values
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — works on headless servers
import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import get_cifar10_loader
from src.hooks import FeatureExtractor, get_stride_layers
from src.models import load_resnet50
from src.spectral import compute_avr, compute_power_spectrum, compute_radial_profile, log_power_spectrum
from src.synthetic import make_checkerboard_image, make_constant_image

FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TABLES_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

STRIDE = 2  # Stride to evaluate Nyquist violations against





def compute_and_report(
    name: str,
    images: torch.Tensor,
    stride: int,
) -> tuple[float, np.ndarray]:
    """Run the full spectral pipeline on *images* and print the AVR.

    Parameters
    name : str
        Human-readable label (used in print output).
    images : Tensor, shape (B, C, H, W)
    stride : int

    Returns
    (avr, power_spectrum) tuple
    """
    ps = compute_power_spectrum(images)
    rp = compute_radial_profile(ps)  # Kept for Step 7 visualization
    avr = compute_avr(ps, stride=stride)
    print(f"  {name:25s}  AVR (stride={stride}) = {avr:.4f}")
    return avr, ps



def plot_power_spectrum(ps: np.ndarray, title: str, filepath: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))

    # Log compress to reduce dynamic range.
    log_ps = np.log1p(ps)

    # Normalize to [0, 1] so matplotlib always uses the full colormap range.
    # Without this, a checkerboard spectrum (4 bright corner pixels, everything
    # else zero) renders as an entirely black image because the colormap scale
    # is set by the max value and everything else is too small to show up.
    ps_min = log_ps.min()
    ps_max = log_ps.max()
    if ps_max - ps_min > 1e-10:
        display = (log_ps - ps_min) / (ps_max - ps_min)
    else:
        display = log_ps

    ax.imshow(display, cmap="inferno", origin="upper", interpolation="nearest")
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(filepath, dpi=100)
    plt.close(fig)
    print(f"  Saved: {filepath}")


def plot_avr_comparison(
    labels: list[str],
    avrs: list[float],
    filepath: Path,
) -> None:
    """Bar chart comparing AVR values.

    Parameters
    labels : list[str]
    avrs : list[float]
    filepath : Path
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["steelblue", "tomato"]
    bars = ax.bar(labels, avrs, color=colors, width=0.5, edgecolor="black")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AVR (Aliasing-Vulnerability Ratio)")
    ax.set_title(f"Phase 1 — AVR Comparison (stride={STRIDE})")
    ax.axhline(0.80, color="red", linestyle="--", linewidth=1, label="0.80 threshold")
    ax.axhline(0.05, color="green", linestyle="--", linewidth=1, label="0.05 threshold")
    for bar, v in zip(bars, avrs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
    ax.legend()
    fig.tight_layout()
    fig.savefig(filepath, dpi=100)
    plt.close(fig)
    print(f"  Saved: {filepath}")



def main() -> None:
    """Run Phase 1 validation and save three diagnostic figures."""
    print("=" * 60)
    print("Phase 1 — Spectral Analysis Validation")
    print("=" * 60)

    H, W = 64, 64
    constant_imgs = make_constant_image(height=H, width=W, batch_size=4)
    checker_imgs = make_checkerboard_image(height=H, width=W, batch_size=4)

    print(f"\nInput shape : {constant_imgs.shape}")
    print(f"Stride under test : {STRIDE}")
    print()

    print("[AVR Results]")
    const_avr, const_ps = compute_and_report("Constant image", constant_imgs, STRIDE)
    check_avr, check_ps = compute_and_report("Checkerboard", checker_imgs, STRIDE)

    print()
    print("[Validation]")
    passed = True

    if const_avr < 0.05:
        print(f"  PASS  Constant image AVR = {const_avr:.4f} < 0.05")
    else:
        print(f"  FAIL  Constant image AVR = {const_avr:.4f} (must be < 0.05)")
        passed = False

    if check_avr > 0.80:
        print(f"  PASS  Checkerboard AVR   = {check_avr:.4f} > 0.80")
    else:
        print(f"  FAIL  Checkerboard AVR   = {check_avr:.4f} (must be > 0.80)")
        passed = False

    print()
    print("[Saving figures]")
    plot_power_spectrum(
        const_ps,
        "Constant Image — Power Spectrum",
        FIGURES_DIR / "constant_power_spectrum.png",
    )
    # Note: the checkerboard power spectrum intentionally looks mostly black.
    # All spectral energy (1.677e+07) is concentrated in a single corner pixel
    # at (0,0) after fftshift -- this is physically correct for a maximum-frequency
    # checkerboard signal. The AVR=1.0 confirms the math is right. The visual
    # is uninformative for this synthetic extreme case but that is expected.
    plot_power_spectrum(
        check_ps,
        "Checkerboard — Power Spectrum",
        FIGURES_DIR / "checkerboard_power_spectrum.png",
    )
    plot_avr_comparison(
        ["Constant", "Checkerboard"],
        [const_avr, check_avr],
        FIGURES_DIR / "avr_comparison.png",
    )
    print(f"\n  Three figures saved to {FIGURES_DIR}")

    print()
    print("[Step 1 — Identify Stride Layers]")
    # We need to know exactly which layers in the model actually perform a stride > 1.
    # We only care about stride=2 for this experiment because that's where the nyquist
    # violations occur. If a layer just has stride=1, it isn't downsampling and won't alias.
    model = load_resnet50(pretrained=True)
    all_stride_layers = get_stride_layers(model)
    print("  [Stride Layers]")
    hook_points = []
    for name, mod, stride in all_stride_layers:
        print(f"    {name:25s} stride={stride:<3d} type={mod.__class__.__name__}")
        if stride == 2:
            hook_points.append(name)

    print(f"\n  Found {len(hook_points)} stride-2 layers.")
    if len(hook_points) < 4:
        # A standard ResNet50 should have exactly 4 stride-2 spots (the stem, plus the
        # first bottleneck block in layer2, layer3, and layer4). If we see less, we probably
        # loaded the wrong model or the architecture changed.
        print("  FAIL  Not enough stride-2 layers detected.")
        sys.exit(1)

    print()
    print("[Step 2 — Load CIFAR-10 Image]")
    # CIFAR-10 is quick to download and standard enough that the network will "understand" it.
    # The loader automatically upsamples the tiny 32x32 image to 224x224 so the ResNet sees
    # the spatial scale it was trained on.
    loader = get_cifar10_loader(n_images=1, batch_size=1)
    batch, _ = next(iter(loader))
    print(f"  Image shape: {batch.shape}")

    print()
    print("[Step 3 — Register Hooks & Forward Pass]")
    # Here is where the magic happens. We attach listeners (hooks) to the layers we just found.
    # capture_input=True is critical: we want the feature map *before* the stride throws away
    # half the data, so we can see how much high-frequency energy was about to be aliased.
    extractor = FeatureExtractor(model, layer_names=hook_points, capture_input=True)
    _ = extractor(batch)

    pre_stride = extractor.pre_stride
    all_hooks_fired = True
    if not pre_stride:
        print("  FAIL  No hooks fired.")
        sys.exit(1)

    print("  [Pre-stride Feature Maps]")
    for name in hook_points:
        if name not in pre_stride:
            print(f"    {name:25s} FAIL (no output)")
            all_hooks_fired = False
        else:
            print(f"    {name:25s} pre_stride shape : {pre_stride[name].shape}")

    if not all_hooks_fired:
        # If a hook didn't fire, the layer name might be wrong or the forward pass bypassed it.
        # We can't proceed if we're missing data.
        print("  FAIL  Missing activations.")
        sys.exit(1)

    print()
    print("[Step 4 — Per-layer AVR]")
    print("  [Per-layer AVR — ResNet50 on CIFAR-10 image]")

    # We will compute the spectral pipeline for every hooked layer.
    # We save all intermediate artifacts (power spectrum, radial profile) in a dict
    # so we don't have to recompute them when generating figures below.
    layer_stats = {}
    for name in hook_points:
        ps = compute_power_spectrum(pre_stride[name])
        rp = compute_radial_profile(ps)  # Kept for Step 7 visualization
        avr = compute_avr(ps, stride=2)
        print(f"    {name:25s} AVR = {avr:.4f}")
        layer_stats[name] = {"ps": ps, "rp": rp, "avr": avr}

    csv_path = TABLES_DIR / "phase1_layerwise_avr.csv"
    with open(csv_path, "w", newline="") as f:
        # Exporting strictly as CSV helps if we ever want to do programmatic analysis
        # rather than just scrolling through terminal output.
        writer = csv.writer(f)
        writer.writerow(["layer_name", "avr"])
        for name in hook_points:
            writer.writerow([name, f"{layer_stats[name]['avr']:.6f}"])
    print(f"  Saved AVRs to {csv_path}")

    print()
    print("[Step 5 — Spectrum Verification]")
    print("  [Spectrum Check]")
    # Natural images (and their feature maps) have a 1/f spectral dropoff, meaning almost
    # all energy lives at low frequencies (near DC). If our pipeline produced a spectrum
    # where the high frequencies (edges) had more energy than the low frequencies (centre),
    # there is a 99% chance our FFT was shifted incorrectly.
    all_spectra_valid = True
    for name in hook_points:
        rp = layer_stats[name]["rp"]
        R = len(rp)
        r_10 = max(1, int(0.1 * R))
        dc_energy = rp[:r_10].sum()
        edge_energy = rp[-r_10:].sum()
        if dc_energy > edge_energy:
            print(f"    {name:25s} DC energy > edge energy : PASS")
        else:
            print(f"    {name:25s} DC energy > edge energy : FAIL")
            all_spectra_valid = False

    print()
    print("[Step 6 — Fig 1: Heatmaps]")
    n_layers = len(hook_points)
    # Give each subplot plenty of vertical room (4 inches per layer) so the labels don't crash.
    fig1, axes1 = plt.subplots(n_layers, 1, figsize=(6, 4 * n_layers))
    if n_layers == 1:
        axes1 = [axes1]

    for ax, name in zip(axes1, hook_points):
        ps = layer_stats[name]["ps"]
        avr = layer_stats[name]["avr"]
        # log_power_spectrum allows us to actually see the high frequencies.
        # Without log, the DC component is so overwhelmingly bright that the rest is pitch black.
        im = ax.imshow(
            log_power_spectrum(ps), cmap="inferno", origin="upper", interpolation="nearest"
        )
        ax.set_title(f"{name} | AVR={avr:.4f}", fontsize=12)
        ax.axis("off")
        fig1.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig1.tight_layout()
    fig1_path = FIGURES_DIR / "fig1_layerwise_power_spectra.png"
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"  Saved {fig1_path}")

    print()
    print("[Step 7 — Fig 2: Radial Profiles]")
    fig2, axes2 = plt.subplots(n_layers, 1, figsize=(7, 4 * n_layers))
    if n_layers == 1:
        axes2 = [axes2]

    for ax, name in zip(axes2, hook_points):
        rp = layer_stats[name]["rp"]
        norm_freq = np.linspace(0, 1, len(rp))
        norm_energy = rp / rp.sum()
        
        ax.plot(norm_freq, norm_energy, color="blue", linewidth=1.5)
        # The Nyquist cutoff is always exactly in the middle of the spectrum for stride=2.
        # Anything to the right of this red line gets irrevocably aliased back into the low frequencies.
        ax.axvline(0.5, color="red", linestyle="--", label="Nyquist cutoff (stride-2)")
        
        ax.set_title(name, fontsize=12)
        ax.set_xlabel("Normalized Frequency")
        ax.set_ylabel("Normalized Energy")
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2_path = FIGURES_DIR / "fig2_radial_profiles.png"
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"  Saved {fig2_path}")

    print()
    print("[Pass Criteria Summary]")
    
    crit_1 = "PASS" if const_avr < 0.05 else "FAIL"
    crit_2 = "PASS" if check_avr > 0.80 else "FAIL"
    crit_3 = "PASS" if len(hook_points) >= 4 else "FAIL"
    crit_4 = "PASS" if all_hooks_fired else "FAIL"
    crit_5 = "PASS" if fig1_path.exists() else "FAIL"
    crit_6 = "PASS" if fig2_path.exists() else "FAIL"
    
    print(f"  Constant image AVR < 0.05              : {crit_1}")
    print(f"  Checkerboard AVR > 0.80               : {crit_2}")
    print(f"  At least 4 stride-2 layers detected   : {crit_3}")
    print(f"  All pre-stride hooks fired             : {crit_4}")
    print(f"  Power spectrum heatmaps saved (Fig 1) : {crit_5}")
    print(f"  Radial profiles saved (Fig 2)         : {crit_6}")
    
    print()
    all_pass = (crit_1 == "PASS" and crit_2 == "PASS" and 
                crit_3 == "PASS" and crit_4 == "PASS" and 
                crit_5 == "PASS" and crit_6 == "PASS")
                
    if all_pass:
        print("=" * 60)
        print("Phase 1 COMPLETE — all criteria met.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("Phase 1 INCOMPLETE — some criteria failed.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
