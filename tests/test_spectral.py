"""
test_spectral.py

9 tests for the three functions in spectral.py. The rule is: touch
spectral.py, run this suite immediately.

Each test targets a property that is mathematically guaranteed -- not
a heuristic threshold I tuned by hand, but something that must hold
from the definition of the DFT or from the Nyquist theorem. If any of
these fail it means the implementation contradicts the math, not just
that a number is slightly off.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.spectral import compute_avr, compute_power_spectrum, compute_radial_profile


@pytest.fixture()
def constant_map():
    """
    All-ones 64x64 tensor. Represents a spatially uniform signal whose
    entire spectral energy sits at DC (f=0). Any test that uses this
    fixture is checking what happens at the low-frequency extreme.
    """
    return torch.ones(64, 64)


@pytest.fixture()
def checkerboard_map():
    """
    64x64 checkerboard in {+1, -1}. Represents a signal at the Nyquist
    limit (f=0.5). Any test that uses this is checking the high-frequency,
    high-AVR extreme. This is also the fixture that broke everything when
    the radial profile was clipped to the inscribed circle.
    """
    row = torch.arange(64).unsqueeze(1)
    col = torch.arange(64).unsqueeze(0)
    board = ((row + col) % 2) * 2.0 - 1.0
    return board.float()


@pytest.fixture()
def zeros_map():
    """
    All-zeros 64x64 tensor. Useful for testing numerical stability (epsilon
    in compute_avr). Not currently used in a test but kept for future checks.
    """
    return torch.zeros(64, 64)


def test_power_spectrum_shape(constant_map):
    """
    Physical property: the 2-D DFT of an NxM input produces an NxM output
    (same spatial grid, different domain). If the shape changes it means
    fft2 or the fftshift call is doing something unexpected -- possibly
    padding is enabled or we are accidentally feeding the wrong array.
    A shape mismatch would silently corrupt all downstream radius calculations.
    """
    ps = compute_power_spectrum(constant_map)
    assert ps.shape == (64, 64), f"Expected (64,64), got {ps.shape}"


def test_power_spectrum_nonnegative(checkerboard_map):
    """
    Physical property: power = |F|^2, and squaring a complex number always
    gives a non-negative real result. If any value is negative it means
    the implementation is returning amplitude or real-part instead of
    squared magnitude, which would make the AVR formula meaningless (you
    cannot threshold negative energy against Nyquist).
    """
    ps = compute_power_spectrum(checkerboard_map)
    assert np.all(ps >= 0), "Power spectrum has negative values"


def test_power_spectrum_constant_is_dc_only(constant_map):
    """
    Physical property: the DFT of a constant f(x,y) = c is a scaled Dirac
    delta at the origin (DC bin). After fftshift, DC is at (H//2, W//2).
    More than 99% of power should be at that single pixel.

    If this fails it means either the fftshift is wrong (DC is not centred)
    or there is spectral leakage from the FFT computation itself. Either way
    the radial profile would be picking up energy at nonzero radii when
    there should be none, which would inflate every AVR measurement.
    """
    ps = compute_power_spectrum(constant_map)
    H, W = ps.shape
    dc_power = ps[H // 2, W // 2]
    total_power = ps.sum()
    assert dc_power / (total_power + 1e-10) > 0.99, (
        f"DC fraction = {dc_power / total_power:.4f}, expected > 0.99"
    )


def test_radial_profile_length(constant_map):
    """
    Implementation contract: the profile must extend to the full diagonal
    of the FFT array, which has length ceil(sqrt((H/2)^2 + (W/2)^2)) + 1.

    This test exists because of the bug I introduced initially by using
    min(H,W)//2 as max_r. That shorter length silently dropped all the
    checkerboard energy (which lives at the corners, radius ~45 for a 64x64
    image) and made AVR return 0. If this test fails, the clipping bug is
    back and test_avr_checkerboard_is_high will also fail.
    """
    import math
    ps = compute_power_spectrum(constant_map)
    rp = compute_radial_profile(ps)
    H, W = 64, 64
    expected_len = math.ceil(math.sqrt((H / 2) ** 2 + (W / 2) ** 2)) + 1
    assert len(rp) == expected_len, (
        f"Expected length {expected_len}, got {len(rp)}"
    )



def test_radial_profile_nonnegative(checkerboard_map):
    """
    Physical property: the radial profile is an average of power values,
    and power is always >= 0, so the average must be >= 0. Negative values
    would indicate a bug in the accumulation step (e.g. subtraction instead
    of addition, or using np.add.at with the wrong sign). A negative bin
    would cause AVR to go above 1, breaking the ratio interpretation.
    """
    ps = compute_power_spectrum(checkerboard_map)
    rp = compute_radial_profile(ps)
    assert np.all(rp >= 0), "Radial profile has negative values"


def test_radial_profile_constant_peaks_at_zero(constant_map):
    """
    Physical property: for a constant image all the spectral energy is
    at DC (radius 0 in the profile). The DC bin should be the unique
    maximum of the whole profile because there is literally no energy
    anywhere else.

    If this fails the fftshift is wrong (DC is not at the centre of the
    2-D array, so the radius-0 bin gets assigned near-zero power and the
    real DC power ends up at some high-radius bin). That would flip the
    expected AVR values: constant images would look high-frequency and
    high-AVR, which is the opposite of what Nyquist says.
    """
    ps = compute_power_spectrum(constant_map)
    rp = compute_radial_profile(ps)
    assert rp[0] == rp.max(), (
        f"DC bin r=0 is not the peak: r=0={rp[0]:.2e}, max={rp.max():.2e}"
    )


def test_avr_constant_is_low(constant_map):
    """
    Physical property: a constant image has all energy at DC (f=0).
    For stride=2 the Nyquist limit is f=0.25. DC is well below that,
    so AVR must be close to 0. We allow up to 0.05 as tolerance for
    the small amount of power that gets scattered to adjacent radius
    bins by integer rounding in compute_radial_profile.

    If this fails it means compute_avr is measuring the wrong frequency
    range -- either nyquist_idx is computed incorrectly, or the sqrt(2)
    scaling factor in the formula is wrong, or the slice is flipped
    (measuring sub-Nyquist power instead of supra-Nyquist power).
    A wrong nyquist_idx here would make all of Phase 2 and 3 useless
    because every AVR number would be measuring the wrong thing.
    """
    ps = compute_power_spectrum(constant_map)
    rp = compute_radial_profile(ps)
    avr = compute_avr(rp, stride=2)
    assert avr < 0.05, f"Constant AVR = {avr:.4f}, expected < 0.05"


def test_avr_checkerboard_is_high(checkerboard_map):
    """
    Physical property: a checkerboard has all energy at the Nyquist corner
    (f=0.5 per axis). For stride=2, Nyquist drops to f=0.25. Almost all
    of the checkerboard's power is above 0.25, so AVR must be > 0.80.

    This test caught the original radial profile bug: when max_r was set
    to min(H,W)//2=32, the corner pixels at radius ~45 were outside the
    profile and their power was silently discarded. AVR came out as 0.
    Fixing max_r to the full diagonal (ceil(sqrt(32^2+32^2))=46) brought
    AVR to 0.9985. If this test fails again, that clipping bug is back.
    """
    ps = compute_power_spectrum(checkerboard_map)
    rp = compute_radial_profile(ps)
    avr = compute_avr(rp, stride=2)
    assert avr > 0.80, f"Checkerboard AVR = {avr:.4f}, expected > 0.80"


def test_avr_range():
    """
    Physical property: AVR is defined as aliased_power / total_power.
    Both the numerator and denominator are sums of non-negative values
    and the numerator is a subset of the denominator, so the ratio must
    lie in [0, 1]. If it goes outside that range the epsilon guard or the
    slice indices in compute_avr are wrong.

    Using random noise tests this over the full frequency spectrum rather
    than just the two extreme cases. A violation here would mean some
    random inputs produce AVR > 1, which would break any downstream
    aggregation or visualisation that interprets AVR as a probability.
    """
    H, W = 32, 32
    rng = np.random.default_rng(42)
    for _ in range(20):
        fm = torch.tensor(rng.standard_normal((H, W)), dtype=torch.float32)
        ps = compute_power_spectrum(fm)
        rp = compute_radial_profile(ps)
        for stride in (1, 2, 4):
            avr = compute_avr(rp, stride=stride)
            assert 0.0 <= avr <= 1.0, f"AVR = {avr:.4f} out of [0,1] for stride={stride}"
