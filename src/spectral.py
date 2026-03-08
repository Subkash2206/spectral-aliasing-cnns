"""
spectral.py

This is basically the heart of the whole project. Everything else feeds into
or out of the three functions here. The goal is to take a 2-D CNN feature map
and answer one question: how much of its energy lives at frequencies that a
stride-s convolution cannot represent without aliasing?

The Nyquist-Shannon theorem says: if you downsample by stride s, you can only
faithfully represent frequencies up to f = 0.5/s (in normalised units where
1.0 means one cycle per pixel). Anything above that gets folded back down
into lower frequencies -- that is aliasing. We can not un-alias after the
fact, which is why it matters to measure it *before* the strided op.

The pipeline is:
  feature map (H, W)
    -> compute_power_spectrum  : squared FFT magnitudes, real-valued, DC centred
    -> compute_radial_profile  : collapses 2-D spectrum to 1-D by radius
    -> compute_avr             : fraction of power above the Nyquist cutoff
"""

from __future__ import annotations

import numpy as np
import torch


def compute_power_spectrum(feature_map: torch.Tensor) -> np.ndarray:
    """
    We need the power spectrum because raw FFT output is complex -- it has
    phase information we do not care about and that we cannot threshold. By
    squaring the magnitude we get real-valued energy at every frequency bin.
    That is the quantity we can actually compare against a Nyquist cutoff.

    Without the fftshift call the DC component (zero frequency, constant
    offset) sits at the corner of the array instead of the centre. The radial
    distance calculation later assumes DC is at (H/2, W/2), so if we skip the
    shift the entire radius map would be wrong and AVR would be garbage.

    Parameters
    feature_map : torch.Tensor
        Either a single 2-D spatial activation (H, W), or a full batch
        (B, C, H, W). If a batch is provided, the function computes the
        power spectrum for every channel of every sample and returns the
        mean power spectrum. Does not need to be normalised.

    Returns
    numpy.ndarray, shape (H, W), dtype float64
        Power at each frequency bin, with DC at the centre. Every value is
        >= 0 because we squared the magnitude.

    Notes
    No window function is applied. A Hann or Hamming window would reduce
    spectral leakage, but since compute_radial_profile averages over all
    pixels at each radius the leakage gets smoothed out enough for our
    coarse AVR metric. Keeping it un-windowed also makes the checkerboard
    sanity check easy to interpret.
    """
    if feature_map.dim() not in (2, 4):
        raise ValueError(
            f"Expected a 2-D (H,W) or 4-D (B,C,H,W) tensor, got shape {tuple(feature_map.shape)}"
        )

    # Convert to NumPy and float64 immediately to avoid precision loss later.
    arr = feature_map.detach().cpu().numpy().astype(np.float64)
    
    # Compute 2D FFT over the last two axes (spatial dimensions).
    fft = np.fft.fft2(arr, axes=(-2, -1))
    
    # Without fftshift, DC is at the corner -- the radius map would be totally wrong.
    fft_shifted = np.fft.fftshift(fft, axes=(-2, -1))
    
    # Square the magnitude to get power (real, non-negative) instead of amplitude.
    power = np.abs(fft_shifted) ** 2
    
    # If it was a batch, average out the non-spatial dimensions
    if power.ndim == 4:
        power = power.mean(axis=(0, 1))
        
    return power


def compute_radial_profile(power_spectrum: np.ndarray) -> np.ndarray:
    """
    A 2-D power spectrum has a lot of redundancy -- energy is roughly
    isotropic in natural images (no preferred spatial direction), so we can
    summarise it by averaging all pixels at the same distance from DC. That
    gives us a 1-D curve: power as a function of spatial frequency magnitude.

    The critical design choice here is how far out we extend the profile.
    My first attempt used min(H,W)//2 as the maximum radius, which is the
    radius of the largest circle that fits inside the array. That completely
    broke the checkerboard test -- all the checkerboard energy sits at the
    *corners* of the shifted FFT (radius ~ H/2 * sqrt(2) for a square image)
    because the checkerboard is the 2-D Dirac at the Nyquist corner. Those
    corners were outside the inscribed circle so all that power was silently
    dropped and AVR came out as zero. Extending max_r to the full diagonal
    fixes it.

    Parameters
    power_spectrum : numpy.ndarray, shape (H, W)
        Output of compute_power_spectrum -- DC at centre, all non-negative.

    Returns
    numpy.ndarray, shape (R,), dtype float64
        Radial-average power. Index r is the mean power of all pixels at
        distance r (in pixels) from DC. R = ceil(sqrt((H/2)^2 + (W/2)^2)) + 1,
        which is large enough to include the four corners of the array.
        Bins that contain no pixels will be zero.
    """
    if power_spectrum.ndim != 2:
        raise ValueError(
            f"Expected 2-D array, got shape {power_spectrum.shape}"
        )

    H, W = power_spectrum.shape
    cy, cx = H // 2, W // 2

    # For every pixel, compute its integer distance from the DC centre.
    # np.indices gives us coordinate grids cheaply.
    y_idx, x_idx = np.indices((H, W))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)

    # Extend to the full diagonal so the corners (where checkerboard energy
    # lives) get captured. The diagonal half-length is sqrt((H/2)^2 + (W/2)^2).
    max_r = int(np.ceil(np.sqrt((H / 2) ** 2 + (W / 2) ** 2)))
    profile = np.zeros(max_r + 1, dtype=np.float64)
    counts = np.zeros(max_r + 1, dtype=np.int64)

    # np.add.at is the non-buffered scatter-add -- needed because multiple
    # pixels share the same integer radius bin and we want to accumulate them.
    np.add.at(profile, r.ravel(), power_spectrum.ravel())
    np.add.at(counts, r.ravel(), 1)

    # Divide accumulated power by the number of pixels in each bin to get
    # a proper average. Guard against empty bins (should not happen in practice
    # for any reasonable image size, but better safe than a divide-by-zero).
    nonzero = counts > 0
    profile[nonzero] /= counts[nonzero]

    return profile


def compute_avr(
    power_spectrum: np.ndarray,
    stride: int,
    epsilon: float = 1e-10,
) -> float:
    """
    Computes AVR directly on the 2D power spectrum, exactly as specified
    in the research bible:

        r_norm(u,v) = sqrt(((u - cy)/cy)^2 + ((v - cx)/cx)^2)
        AVR = sum of P(u,v) where r_norm > 1/stride, divided by total power

    r_norm is normalized so that r_norm=1 at the per-axis Nyquist edge
    (the halfway point along each axis of the frequency grid). This means:
      - DC at center has r_norm = 0
      - Per-axis Nyquist edge has r_norm = 1
      - Diagonal corners have r_norm = sqrt(2)

    For stride=2, the cutoff is r_norm=0.5. Anything above that aliases.

    This formulation avoids the sqrt(2) error in the radial profile approach,
    where using the diagonal as the normalization reference placed the cutoff
    1.41x too far out and caused AVR to be underestimated.

    Parameters
    power_spectrum : np.ndarray, shape (H, W)
        Output of compute_power_spectrum -- DC at center, all non-negative.
    stride : int
        Downsampling factor of the operation following this feature map.
    epsilon : float
        Added to denominator to avoid divide-by-zero on blank feature maps.

    Returns
    float in [0, 1]. 0 = fully bandlimited. 1 = all energy above Nyquist.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    H, W = power_spectrum.shape
    cy, cx = H // 2, W // 2

    y_idx, x_idx = np.indices((H, W))

    # Normalize so r_norm=1 at the per-axis Nyquist edge.
    # Dividing by cy and cx (the half-dimensions) achieves this.
    # The diagonal corners end up at r_norm = sqrt(2), which is above
    # any reasonable stride cutoff and correctly counted as aliased.
    r_norm = np.sqrt(((y_idx - cy) / cy) ** 2 + ((x_idx - cx) / cx) ** 2)

    # Per the bible: cutoff is at r_norm = 1/stride.
    r_cutoff = 1.0 / stride

    total_power = float(power_spectrum.sum()) + epsilon
    aliased_power = float(power_spectrum[r_norm > r_cutoff].sum())

    return aliased_power / total_power


def log_power_spectrum(power_spectrum: np.ndarray) -> np.ndarray:
    """
    Applies a log(1 + x) compression to the power spectrum before plotting.

    We need this because the DC component of a real image is typically
    orders of magnitude larger than the high-frequency components. Without
    compression, a heatmap of the raw power spectrum would show one bright
    dot at the centre and everything else as pitch black -- completely
    uninformative. log1p brings the dynamic range to something the eye can
    actually distinguish.

    log1p (instead of log) avoids the undefined log(0) for zero-energy bins.

    Parameters
    power_spectrum : numpy.ndarray, shape (H, W)
        Output of compute_power_spectrum.

    Returns
    numpy.ndarray, shape (H, W), dtype float64
        Log-compressed power, all values >= 0.
    """
    return np.log1p(power_spectrum)

