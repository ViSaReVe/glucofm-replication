"""Appendix C.7 training corruptions; invoked afresh for each training batch."""

import math
import torch


def compression_profile(length: int, bottom: float, device=None, dtype=torch.float32):
    """C.7 compression envelope: a V from 1 down to exactly `bottom` and back.

    `linspace(-1, 1, length).abs()` contains an exact zero only for odd lengths, so
    an unrenormalised even-length ramp bottoms out above the sampled depth (length 6,
    bottom 0.40 reaches only 0.52). Rescaling the ramp to span [0, 1] makes the
    sampled nadir attained at every length and leaves odd lengths unchanged.
    """
    if length < 3:
        raise ValueError("A compression envelope needs at least three positions")
    ramp = torch.linspace(-1, 1, length, device=device, dtype=dtype).abs()
    ramp = (ramp - ramp.min()) / (1 - ramp.min())
    return bottom + (1 - bottom) * ramp


@torch.no_grad()
def augment(glucose, observed):
    x, mask = glucose.clone(), observed.clone()
    device = x.device
    steps = x.shape[1]
    time = torch.arange(steps, device=device, dtype=x.dtype) / steps
    for row in range(len(x)):
        decay = 1.0
        for kind in torch.randperm(4).tolist():
            probability = (0.25, 0.10, 0.40, 0.05)[kind] * decay
            if kind == 2 and mask[row].sum() <= 200:
                continue
            if torch.rand(()).item() >= probability:
                continue
            if kind == 0:
                amplitude = 5 + 10 * torch.rand(()).item()
                frequency = 0.5 + 1.5 * torch.rand(()).item()
                phase = 2 * math.pi * torch.rand(()).item()
                x[row] += amplitude * torch.sin(2 * math.pi * frequency * time + phase)
            elif kind == 1:
                length = int(torch.randint(6, 13, ()).item())
                start = int(torch.randint(0, steps-length+1, ()).item())
                bottom = 0.4 + 0.3 * torch.rand(()).item()
                x[row, start:start+length] *= compression_profile(length, bottom, device, x.dtype)
            elif kind == 2:
                offset = int(torch.randint(0, 3, ()).item())
                mask[row] &= torch.arange(steps, device=device) % 3 == offset
            else:
                for _ in range(int(torch.randint(1, 4, ()).item())):
                    length = int(torch.randint(2, 13, ()).item())
                    start = int(torch.randint(0, steps-length+1, ()).item())
                    mask[row, start:start+length] = False
            decay *= 0.25
        # Degenerate sparse windows must not become empty after corruption.
        if not mask[row].any():
            mask[row] = observed[row]
            x[row] = glucose[row]
    return torch.where(mask, x, 0.0), mask
