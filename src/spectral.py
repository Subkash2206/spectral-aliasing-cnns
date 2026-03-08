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
        A single 2-D spatial activation, shape (H, W). One channel of one
        sample. Does not need to be normalised -- we only care about the
        relative distribution of power across frequencies.

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
    if feature_map.dim() != 2:
        raise ValueError(
            f"Expected a 2-D tensor (H, W), got shape {tuple(feature_map.shape)}"
        )

    arr = feature_map.float().numpy()
    fft = np.fft.fft2(arr)
    # Without fftshift, DC is at the corner -- the radius map would be totally wrong.
    fft_shifted = np.fft.fftshift(fft)
    # Square the magnitude to get power (real, non-negative) instead of amplitude.
    power = np.abs(fft_shifted) ** 2
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
    radial_profile: np.ndarray,
    stride: int,
    epsilon: float = 1e-10,
) -> float:
    """
    AVR (Aliasing-Vulnerability Ratio) is the core metric of this project.
    It answers: what fraction of a feature map's power would get aliased if
    the next layer downsamples with stride s?

    Nyquist says the highest frequency you can represent after stride-s
    downsampling is f_nyquist = 0.5 / s (per axis, where 0.5 = 1 cycle per
    2 pixels = the Nyquist limit of the original grid). Anything above that
    gets folded back and corrupts lower frequencies.

    The tricky bit is converting that frequency cutoff to a pixel radius in
    our radial profile. The profile goes from r=0 (DC) to r=R-1 (the corner
    of the FFT array). The corner sits at the 2-D Nyquist corner, which has
    per-axis frequency 0.5 but a diagonal frequency of sqrt(0.5^2 + 0.5^2)
    = 0.5*sqrt(2). So:

        corner radius R-1  <->  per-axis frequency 0.5
        nyquist radius     <->  per-axis frequency 0.5/stride

    Dividing: nyquist_idx = (R-1) / (stride * sqrt(2))

    Everything at radius >= nyquist_idx is above the Nyquist cutoff.
    AVR = sum(profile[nyquist_idx:]) / sum(profile).

    Parameters
    radial_profile : numpy.ndarray, shape (R,)
        Output of compute_radial_profile.
    stride : int
        The stride of the convolution that comes *after* this feature map.
        Must be >= 1. stride=1 means no downsampling (Nyquist = 0.5, AVR
        will be high only if the signal has near-Nyquist content).
    epsilon : float, optional
        Added to the denominator so we never divide by zero on a blank
        feature map (it does happen with zero-initialised networks).

    Returns
    float
        AVR in [0, 1]. 0 means all power is DC or sub-Nyquist (great).
        1 means all power is above the Nyquist cutoff (maximally vulnerable).

    Raises
    ValueError
        If stride < 1 or the profile is empty. Both would give nonsense
        results and I want to catch them explicitly rather than get a
        confusing divide-by-zero or silent wrong answer downstream.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if radial_profile.size == 0:
        raise ValueError("radial_profile must not be empty")

    R = len(radial_profile)
    # Map per-axis Nyquist to a pixel radius index.
    # (R-1) is the diagonal corner = frequency sqrt(2)/2 along the diagonal.
    # The per-axis Nyquist (0.5/stride) corresponds to pixel radius:
    #   nyquist_r = (R-1) / (stride * sqrt(2))
    nyquist_idx = int((R - 1) / (stride * np.sqrt(2)))
    # Clamp to valid range just in case of floating-point weirdness.
    nyquist_idx = max(0, min(nyquist_idx, R - 1))

    total_power = float(radial_profile.sum()) + epsilon
    aliased_power = float(radial_profile[nyquist_idx:].sum())

    # AVR = fraction of power sitting above the Nyquist cutoff.
    return aliased_power / total_power
