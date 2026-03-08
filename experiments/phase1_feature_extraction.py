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

from src.spectral import compute_avr, compute_power_spectrum, compute_radial_profile
from src.synthetic import make_checkerboard_image, make_constant_image

FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

STRIDE = 2  # Stride to evaluate Nyquist violations against



def single_channel_2d(batch: torch.Tensor) -> torch.Tensor:
    """Extract the first (sample, channel) 2-D feature map from a batch.

    Parameters
    batch : Tensor, shape (B, C, H, W)

    Returns
    Tensor, shape (H, W)
    """
    return batch[0, 0]


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
    fm = single_channel_2d(images)  # (H, W)
    ps = compute_power_spectrum(fm)
    rp = compute_radial_profile(ps)
    avr = compute_avr(rp, stride=stride)
    print(f"  {name:25s}  AVR (stride={stride}) = {avr:.4f}")
    return avr, ps



def plot_power_spectrum(ps: np.ndarray, title: str, filepath: Path) -> None:
    """Save a log-scale 2-D power spectrum plot.

    Parameters
    ps : ndarray, shape (H, W)
        Power spectrum (DC centred).
    title : str
        Plot title.
    filepath : Path
        Destination file path.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(
        np.log1p(ps),
        cmap="inferno",
        origin="upper",
        interpolation="nearest",
    )
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
    if passed:
        print("=" * 60)
        print("Phase 1 COMPLETE — all criteria met.")
        print("=" * 60)
    else:
        print("=" * 60)
        print("Phase 1 FAILED — see FAIL lines above.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
