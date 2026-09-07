"""Appendix C.7 training corruptions; invoked afresh for each training batch."""

import math
import torch


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
                v = torch.linspace(-1, 1, length, device=device).abs()
                x[row, start:start+length] *= bottom + (1-bottom) * v
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
