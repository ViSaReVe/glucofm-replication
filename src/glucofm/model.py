"""GlucoFM v1 architecture and objectives (paper §3, Appendix C).

Local stream tokens are fused BEFORE contextual attention. The context mask is
applied BEFORE all input-derived statistics. Underspecified details are recorded
in docs/implementation-decisions.md rather than attributed to the paper.
"""

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

STEPS, PATCH, PATCHES = 288, 12, 24


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 3
    heads: int = 4
    width: int = 128
    feedforward: int = 256
    dropout: float = 0.1  # Implementation assumption; not specified in v1.

    def __post_init__(self):
        if self.width != 128 or self.heads != 4 or self.feedforward != 256:
            raise ValueError("This implementation fixes the Appendix C.4 dimensions.")
        if self.layers < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid layer count or dropout.")


def masked_moments(x: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Last-axis mean/std; empty entries are zero, including NaN placeholders."""
    clean = torch.where(valid, x, 0.0)
    count = valid.sum(-1, keepdim=True).clamp_min(1)
    mean = clean.sum(-1, keepdim=True) / count
    variance = torch.where(valid, (clean - mean).square(), 0.0).sum(-1, keepdim=True) / count
    # clamp avoids infinite sqrt derivatives for constant or empty patches.
    std = variance.clamp_min(1e-8).sqrt()
    support = valid.any(-1, keepdim=True)
    return torch.where(support, mean, 0.0), torch.where(support, std, 0.0)


def normalize_visible(x: Tensor, valid: Tensor) -> Tensor:
    """Per-window observed mean/std (explicit assumption), minimum scale 1 mg/dL."""
    mean, std = masked_moments(x, valid)
    return torch.where(valid, (torch.where(valid, x, 0.0) - mean) / std.clamp_min(1.0), 0.0)


def rate_of_change(x: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
    """Eq. 10: closest preceding observation, up to 9 five-minute steps away."""
    clean = torch.where(valid, x, 0.0)
    rate = torch.zeros_like(clean)
    found = torch.zeros_like(valid)
    for lag in range(1, min(10, x.shape[-1])):
        previous = F.pad(clean[:, :-lag], (lag, 0))
        previous_valid = F.pad(valid[:, :-lag], (lag, 0), value=False)
        pair = valid & previous_valid & ~found
        rate = torch.where(pair, (clean - previous) / lag, rate)
        found = found | pair
    return rate, found


class CausalGaussian(nn.Module):
    """Eqs. 11–14. Only this filtering operation is claimed to be causal."""

    def __init__(self):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(math.log(0.4 / 0.6)))
        self.register_buffer("lags", torch.arange(37, dtype=torch.float32))

    @property
    def sigma(self) -> Tensor:
        return 2.0 + 10.0 * self.rho.sigmoid()

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        weights = torch.exp(-self.lags.square() / (2 * self.sigma.square()))
        weights = weights / weights.sum()
        # conv1d computes correlation: flip so lag zero multiplies current input.
        kernel = weights.flip(0).view(1, 1, -1)
        clean = torch.where(valid, x, 0.0)
        numerator = F.conv1d(F.pad(clean[:, None], (36, 0)), kernel)[:, 0]
        denominator = F.conv1d(F.pad(valid[:, None].to(x.dtype), (36, 0)), kernel)[:, 0]
        return numerator / denominator.clamp_min(1e-8)


class WaveFeature(nn.Module):
    """Within-patch convolution, activation, and observed-position pooling."""

    def __init__(self, width: int):
        super().__init__()
        self.conv = nn.Conv1d(1, width, kernel_size=3, padding=1)

    def forward(self, patches: Tensor, valid: Tensor) -> Tensor:
        b, p, k = patches.shape
        x = torch.where(valid, patches, 0.0).reshape(b * p, 1, k)
        features = F.gelu(self.conv(x))
        weights = valid.reshape(b * p, 1, k).to(x.dtype)
        pooled = (features * weights).sum(-1) / weights.sum(-1).clamp_min(1)
        return pooled.reshape(b, p, -1)


class StreamEmbedder(nn.Module):
    def __init__(self, waveform_width: int, difference_width: int, statistics_width: int):
        super().__init__()
        self.wave = WaveFeature(waveform_width)
        self.difference = WaveFeature(difference_width)
        self.statistics = nn.Sequential(
            nn.Linear(2, statistics_width), nn.GELU(), nn.Linear(statistics_width, statistics_width)
        )
        self.projection = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.LayerNorm(64))

    def forward(self, waveform, difference, stats, valid, difference_valid):
        features = torch.cat([
            self.wave(waveform, valid),
            self.difference(difference, difference_valid),
            self.statistics(stats),
        ], dim=-1)
        return self.projection(features) * valid.any(-1, keepdim=True)


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig, layers: int):
        super().__init__()
        # Each layer initialized independently (no cloned identical initialization).
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                config.width, config.heads, config.feedforward, config.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(config.width)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class GlucoFMEncoder(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.filter = CausalGaussian()
        self.state_embedder = StreamEmbedder(64, 16, 48)
        self.event_embedder = StreamEmbedder(48, 48, 32)
        self.fusion = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.LayerNorm(128))
        self.time_projection = nn.Linear(2, 128)
        self.position = nn.Parameter(torch.randn(1, PATCHES, 128) * 0.02)
        self.time_gate = nn.Parameter(torch.zeros(()))
        self.mask_token = nn.Parameter(torch.randn(1, 1, 128) * 0.02)
        self.context = Transformer(self.config, self.config.layers)

    def forward(self, glucose: Tensor, observed: Tensor, start: Tensor,
                hidden: Tensor | None = None) -> dict[str, Tensor]:
        if glucose.ndim != 2 or glucose.shape[1] != STEPS or observed.shape != glucose.shape:
            raise ValueError("Expected glucose and observed arrays of shape [batch, 288].")
        if not glucose.is_floating_point():
            raise ValueError("glucose must be floating point.")
        if observed.dtype != torch.bool:
            raise ValueError("observed must be a boolean physical observation mask.")
        if not torch.isfinite(glucose[observed]).all():
            raise ValueError("Observed glucose must be finite; NaN is allowed only where missing.")
        if not observed.any(-1).all():
            raise ValueError("Each window needs at least one physical observation.")
        b = glucose.shape[0]
        if start.dtype not in (torch.int32, torch.int64) or start.shape != (b,) or ((start < 0) | (start >= STEPS)).any():
            raise ValueError("start must contain one clock index in [0, 288) per window.")
        if hidden is None:
            hidden = torch.zeros(b, PATCHES, dtype=torch.bool, device=glucose.device)
        if hidden.shape != (b, PATCHES) or hidden.dtype != torch.bool:
            raise ValueError("hidden must be a boolean [batch, 24] patch mask.")
        visible = observed & ~hidden.repeat_interleave(PATCH, dim=1)
        raw = torch.where(visible, glucose, 0.0)
        normalized = normalize_visible(raw, visible)
        trend = self.filter(normalized, visible)
        residual = torch.where(visible, normalized - trend, 0.0)
        valid = visible.reshape(b, PATCHES, PATCH)
        state_wave = trend.reshape(b, PATCHES, PATCH)
        event_wave = residual.reshape(b, PATCHES, PATCH)
        differences = F.pad(state_wave[..., 1:] - state_wave[..., :-1], (1, 0))
        difference_valid = F.pad(valid[..., 1:] & valid[..., :-1], (1, 0), value=False)
        mean, std = masked_moments(raw.reshape(b, PATCHES, PATCH), valid)
        state_stats = torch.cat([mean, std], -1) / 100.0
        rate, rate_valid = rate_of_change(raw, visible)
        rate = rate.reshape(b, PATCHES, PATCH)
        rate_valid = rate_valid.reshape(b, PATCHES, PATCH)
        rate_mean, rate_std = masked_moments(rate, rate_valid)
        event_stats = torch.cat([rate_mean, rate_std], -1) / 100.0
        state = self.state_embedder(state_wave, differences, state_stats, valid, difference_valid)
        event = self.event_embedder(event_wave, rate / 100.0, event_stats, valid, rate_valid)
        clock_index = (start[:, None] + torch.arange(PATCHES, device=start.device)[None] * PATCH) % STEPS
        phase = 2 * math.pi * clock_index.to(glucose.dtype) / STEPS
        clock = torch.stack([phase.sin(), phase.cos()], dim=-1)
        fused = self.fusion(torch.cat([state, event], dim=-1))
        fused = torch.where(hidden[..., None], self.mask_token, fused)
        gate = self.time_gate.sigmoid()
        tokens = self.context(fused + gate * self.time_projection(clock) + (1 - gate) * self.position)
        return {
            "embedding": tokens.mean(1), "tokens": tokens,
            "state": state, "event": event, "clock": clock,
            "trend": trend, "residual": residual,
        }


def weighted_smooth_l1(predicted: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    error = F.smooth_l1_loss(predicted, target.detach(), reduction="none").mean(-1)
    return (error * weight).sum() / weight.sum().clamp_min(1e-8)


def transition_weights(observed: Tensor, hidden: Tensor) -> Tensor:
    density = observed.to(torch.float32).reshape(-1, PATCHES, PATCH).mean(-1)
    return (~hidden[:, :-1]).to(density.dtype) * density[:, :-1] * density[:, 1:]


def sample_hidden(batch: int, device: torch.device) -> Tensor:
    ratios = torch.empty(batch, device=device).uniform_(0.5, 0.6)
    counts = (ratios * PATCHES).floor().long()
    ranks = torch.rand(batch, PATCHES, device=device).argsort(-1).argsort(-1)
    return ranks < counts[:, None]


class GlucoFMPretrainer(nn.Module):
    def __init__(self, config: ModelConfig | None = None, momentum: float = 0.997,
                 dynamics_weight: float = 1.0):
        super().__init__()
        if not 0 <= momentum < 1 or dynamics_weight < 0:
            raise ValueError("Invalid EMA momentum or dynamics weight.")
        self.online = GlucoFMEncoder(config)
        self.target = deepcopy(self.online).requires_grad_(False).eval()
        self.predictor = Transformer(self.online.config, 1)
        self.prediction_projection = nn.Linear(128, 128)
        self.state_transition = nn.Sequential(nn.Linear(130, 128), nn.GELU(), nn.Linear(128, 64))
        self.event_transition = nn.Sequential(nn.Linear(130, 128), nn.GELU(), nn.Linear(128, 64))
        self.momentum, self.dynamics_weight = momentum, dynamics_weight

    def train(self, mode=True):
        super().train(mode)
        self.target.eval()  # Teacher dropout must stay disabled even when student trains.
        return self

    @torch.no_grad()
    def update_target(self):
        for target, online in zip(self.target.parameters(), self.online.parameters(), strict=True):
            target.lerp_(online, 1 - self.momentum)
        for target, online in zip(self.target.buffers(), self.online.buffers(), strict=True):
            target.copy_(online)

    def forward(self, glucose, observed, start, hidden=None):
        if hidden is None:
            hidden = sample_hidden(glucose.shape[0], glucose.device)
        online = self.online(glucose, observed, start, hidden)
        with torch.no_grad():
            target = self.target(glucose, observed, start)
        predicted = self.prediction_projection(self.predictor(online["tokens"]))
        density = observed.to(glucose.dtype).reshape(-1, PATCHES, PATCH).mean(-1)
        contextual = weighted_smooth_l1(predicted, target["tokens"], density * hidden)
        s, e, t = online["state"][:, :-1], online["event"][:, :-1], online["clock"][:, :-1]
        state_next = s + self.state_transition(torch.cat([s, e, t], -1))
        event_next = e + self.event_transition(torch.cat([e, s, t], -1))
        q = transition_weights(observed, hidden)
        dynamics = 0.5 * (
            weighted_smooth_l1(state_next, target["state"][:, 1:], q)
            + weighted_smooth_l1(event_next, target["event"][:, 1:], q)
        )
        return {"loss": contextual + self.dynamics_weight * dynamics,
                "contextual": contextual, "dynamics": dynamics, "hidden": hidden}
