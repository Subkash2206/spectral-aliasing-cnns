"""
synthetic.py

To test whether the spectral analysis pipeline is working correctly, I
need inputs whose frequency content I know exactly from first principles.
The two extremes of the frequency spectrum give me perfect test cases:

  - A constant image has zero spatial frequency variation (all energy at
    DC). Its AVR for any stride should be essentially zero. If it is not,
    the power spectrum or AVR formula is wrong.

  - A 1-pixel checkerboard alternates +1 and -1 every pixel, which is
    exactly the Nyquist frequency (0.5 cycles/pixel) -- the highest
    frequency the grid can represent. For stride >= 2, the Nyquist limit
    is 0.25 cycles/pixel, which is below the checkerboard frequency, so
    virtually all the power is above the cutoff and AVR should be close
    to 1. If it is not, same conclusion: something is broken.

These two stimuli bracket everything a real CNN feature map can do. Any
physically reasonable activation should have an AVR somewhere between
these two extremes.
"""

from __future__ import annotations

import torch


def make_constant_image(
    height: int = 64,
    width: int = 64,
    value: float = 1.0,
    batch_size: int = 1,
    channels: int = 3,
) -> torch.Tensor:
    """
    Returns a batch of spatially uniform images. Every pixel is set to
    value, so the 2-D FFT is a single spike at the DC bin (centre of the
    shifted spectrum) and zero everywhere else. This should give AVR ~ 0
    for any stride because there is literally no high-frequency content
    to alias.

    This is the lower bound sanity check. If the constant image gives
    a high AVR, the power spectrum is accumulating energy at non-DC
    bins somehow (numerical issues with fft, wrong fftshift, etc.).

    Parameters
    height, width : int
        Spatial dimensions of each image in the batch.
    value : float
        Uniform pixel value. The actual number does not matter for spectral
        shape, only for the total power scale.
    batch_size : int
        Number of images to return (stacked along dim 0).
    channels : int
        Number of channels per image (3 for RGB).

    Returns
    torch.Tensor, shape (batch_size, channels, height, width)
        Tensor filled uniformly with value.
    """
    return torch.full((batch_size, channels, height, width), value)


def make_checkerboard_image(
    height: int = 64,
    width: int = 64,
    batch_size: int = 1,
    channels: int = 3,
) -> torch.Tensor:
    """
    Returns a batch of 1-pixel alternating checkerboard images with values
    in {+1, -1}. This pattern has all its spectral energy concentrated at
    the four corners of the 2-D FFT (the Nyquist corner frequency), which
    is exactly the highest representable frequency.

    For stride=2, the Nyquist limit drops to 0.25 cycles/pixel, which is
    below 0.5, so essentially all the checkerboard's power is above the
    Nyquist cutoff. AVR should be very close to 1 (I got 0.9985 in practice,
    the small gap is because the radial binning averages a few pixels that
    straddle the cutoff boundary).

    This is the upper bound sanity check. If AVR is not high for a
    checkerboard, the radial profile or AVR formula is clipping the energy
    -- which is exactly the bug I found when I first implemented the
    profile with max_r = min(H,W)//2. The corner radius is ~H/2*sqrt(2),
    well outside that smaller inscribed circle.

    Parameters
    height, width : int
        Spatial dimensions.
    batch_size : int
        Number of copies in the batch. All copies are identical.
    channels : int
        Number of channels (all channels have the same checkerboard).

    Returns
    torch.Tensor, shape (batch_size, channels, height, width)
        Values alternating between +1 and -1 in a checkerboard pattern.
    """
    # Build a 2-D coordinate grid for one channel.
    # row_idx is (H, 1) and col_idx is (1, W) so the addition broadcasts
    # to give us the Manhattan sum at every pixel.
    row_idx = torch.arange(height).unsqueeze(1)  # (H, 1)
    col_idx = torch.arange(width).unsqueeze(0)   # (1, W)
    # (row + col) % 2 is 0 on white squares, 1 on black squares.
    # Multiplying by 2 and subtracting 1 maps that to {-1, +1}.
    board = (((row_idx + col_idx) % 2) * 2 - 1).float()  # {+1, -1}

    # Add batch and channel dims, then expand to the requested shape.
    # .clone() at the end makes the tensor contiguous and not a view,
    # which matters if the caller modifies it in-place later.
    board = board.unsqueeze(0).unsqueeze(0)
    return board.expand(batch_size, channels, height, width).clone()
