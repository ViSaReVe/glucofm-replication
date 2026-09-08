import pytest
import torch

from glucofm.model import (
    CausalGaussian, GlucoFMEncoder, GlucoFMPretrainer, ModelConfig, WaveFeature,
    masked_moments, rate_of_change, sample_hidden, transition_weights, weighted_smooth_l1,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(17)


def batch():
    glucose = 110 + 20 * torch.randn(3, 288)
    observed = torch.ones_like(glucose, dtype=torch.bool)
    observed[:, 40:48] = False
    return glucose, observed, torch.tensor([0, 78, 282])


def test_masked_statistics_ignore_nan_and_filled_values():
    values = torch.tensor([[100.0, 105, float("nan"), 125], [9.0, 8, 7, 6]])
    valid = torch.tensor([[True, True, False, True], [False] * 4])
    mean, std = masked_moments(values, valid)
    torch.testing.assert_close(mean, torch.tensor([[110.0], [0.0]]))
    assert std[1] == 0 and torch.isfinite(std).all()


def test_filter_matches_hand_computed_weighted_mean():
    model = CausalGaussian()
    values = torch.tensor([[100.0, -999, 120]])
    valid = torch.tensor([[True, False, True]])
    result = model(values, valid)
    weight = torch.exp(torch.tensor(-4 / 72))  # lag 2, sigma 6
    expected = (120 + 100 * weight) / (1 + weight)
    torch.testing.assert_close(result[0, 2], expected)
    torch.testing.assert_close(model.sigma, torch.tensor(6.0))


def test_filter_is_causal_with_fixed_normalized_inputs():
    model = CausalGaussian()
    x = torch.randn(2, 288)
    valid = torch.ones_like(x, dtype=torch.bool)
    changed = x.clone()
    changed[:, 101:] += 1000
    torch.testing.assert_close(model(x, valid)[:, :101], model(changed, valid)[:, :101])


def test_filter_ignores_missing_placeholder_and_stays_bounded():
    model = CausalGaussian()
    x = torch.randn(2, 288)
    valid = torch.rand(2, 288) > 0.4
    changed = x.clone()
    changed[~valid] = float("nan")
    torch.testing.assert_close(model(x, valid), model(changed, valid))
    for rho in (-100.0, 100.0):
        with torch.no_grad():
            model.rho.fill_(rho)
        assert 2 <= model.sigma.item() <= 12


def test_rate_respects_elapsed_steps_and_nearest_valid_observation():
    values = torch.tensor([[100., 999., 999., 130., 150.]])
    valid = torch.tensor([[True, False, False, True, True]])
    rate, supported = rate_of_change(values, valid)
    torch.testing.assert_close(rate, torch.tensor([[0., 0., 0., 10., 20.]]))
    assert supported.tolist() == [[False, False, False, True, True]]


def test_hidden_readings_cannot_change_online_features_or_tokens():
    encoder = GlucoFMEncoder(ModelConfig(dropout=0)).eval()
    x, observed, start = batch()
    hidden = torch.zeros(3, 24, dtype=torch.bool)
    hidden[:, 5:15] = True
    changed = x.clone()
    changed[hidden.repeat_interleave(12, 1)] += 10000
    first, second = encoder(x, observed, start, hidden), encoder(changed, observed, start, hidden)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_missing_values_cannot_change_embeddings():
    encoder = GlucoFMEncoder().eval()
    x, observed, start = batch()
    changed = x.clone()
    changed[~observed] = float("nan")
    with torch.no_grad():
        torch.testing.assert_close(encoder(x, observed, start)["embedding"],
                                   encoder(changed, observed, start)["embedding"])


def test_contextual_loss_trains_both_streams_fusion_filter_and_context():
    model = GlucoFMPretrainer(ModelConfig(dropout=0))
    x, observed, start = batch()
    result = model(x, observed, start)
    result["contextual"].backward()
    for module in (model.online.state_embedder, model.online.event_embedder,
                   model.online.fusion, model.online.context, model.predictor):
        total = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
        assert total > 0, type(module).__name__
    assert model.online.filter.rho.grad.abs().item() > 0
    assert model.online.mask_token.grad.abs().sum().item() > 0
    assert all(p.grad is None for p in model.target.parameters())


def test_dynamics_heads_and_both_streams_receive_gradients():
    model = GlucoFMPretrainer(ModelConfig(dropout=0))
    result = model(*batch())
    result["dynamics"].backward()
    for module in (model.state_transition, model.event_transition,
                   model.online.state_embedder, model.online.event_embedder):
        assert sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None) > 0
    # The local transition objective has no path through global attention.
    assert all(p.grad is None for p in model.online.context.parameters())


def test_optimizer_updates_both_streams_and_fusion():
    model = GlucoFMPretrainer(ModelConfig(dropout=0))
    parameters = [model.online.state_embedder.projection[0].weight,
                  model.online.event_embedder.projection[0].weight, model.online.fusion[0].weight]
    before = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    model(*batch())["loss"].backward()
    optimizer.step()
    assert all(not torch.equal(old, new) for old, new in zip(before, parameters))


def test_teacher_is_frozen_eval_and_ema_matches_formula():
    model = GlucoFMPretrainer(momentum=0.9).train()
    assert not model.target.training
    assert all(not p.requires_grad for p in model.target.parameters())
    before = model.target.filter.rho.detach().clone()
    with torch.no_grad():
        model.online.filter.rho.add_(1)
    model.update_target()
    torch.testing.assert_close(model.target.filter.rho, before + 0.1)


def test_transition_weight_excludes_hidden_source_but_not_hidden_target():
    observed = torch.zeros(1, 288, dtype=torch.bool)
    observed[:, :12] = True
    observed[:, 12:18] = True
    observed[:, 24:36] = True
    hidden = torch.zeros(1, 24, dtype=torch.bool)
    hidden[:, 1] = True
    result = transition_weights(observed, hidden)
    assert result[0, 0] == 0.5 and result[0, 1] == 0 and result[0, 2] == 0


def test_zero_weight_targets_do_not_affect_loss():
    predicted = torch.randn(2, 3, 8, requires_grad=True)
    target = torch.randn_like(predicted)
    weights = torch.tensor([[1., 0., 0.5], [0., 0., 1.]])
    changed = target.clone()
    changed[weights == 0] += 10000
    torch.testing.assert_close(weighted_smooth_l1(predicted, target, weights),
                               weighted_smooth_l1(predicted, changed, weights))
    zero = weighted_smooth_l1(predicted, target, torch.zeros_like(weights))
    zero.backward()
    assert zero == 0 and predicted.grad.abs().sum() == 0


def test_shapes_clock_and_daily_embedding():
    model = GlucoFMEncoder().eval()
    x, observed, start = batch()
    output = model(x, observed, start)
    assert output["state"].shape == (3, 24, 64)
    assert output["event"].shape == (3, 24, 64)
    assert output["tokens"].shape == (3, 24, 128)
    torch.testing.assert_close(output["embedding"], output["tokens"].mean(1))
    torch.testing.assert_close(output["clock"][0, 0], torch.tensor([0., 1.]))
    shifted = model(x, observed, (start + 72) % 288)
    assert not torch.allclose(output["embedding"], shifted["embedding"])


def test_reject_empty_physical_window_but_handle_empty_patches():
    encoder = GlucoFMEncoder()
    x, observed, start = batch()
    observed[:, :120] = False
    assert torch.isfinite(encoder(x, observed, start)["embedding"]).all()
    observed[0] = False
    with pytest.raises(ValueError, match="physical observation"):
        encoder(x, observed, start)


def test_mask_counts_match_documented_flooring():
    hidden = sample_hidden(100, torch.device("cpu"))
    assert set(hidden.sum(1).tolist()) <= {12, 13, 14}


def gap_pair():
    """Two patches with identical zero-filled values but gaps in different places.

    Both hold the constant 2.0 with a 0.0 at positions 3 and 5; patch A observes the
    0.0 at 5 and is missing at 3, patch B is the other way round. The zero-filled
    value arrays are therefore bit-identical, and because both zeros sit in an
    identical (2, 0, 2) neighbourhood the kernel response at 3 equals the response at
    5, so the masked mean coincides too: under a values-only convolution these two
    patches are indistinguishable.
    """
    values = torch.full((1, 1, 12), 2.0)
    values[0, 0, 3] = 0.0
    values[0, 0, 5] = 0.0
    first = torch.ones(1, 1, 12, dtype=torch.bool)
    first[0, 0, 3] = False
    second = torch.ones(1, 1, 12, dtype=torch.bool)
    second[0, 0, 5] = False
    return values, first, second


def test_zero_filling_alone_cannot_separate_a_gap_from_a_mid_range_reading():
    # The premise of the mask channel: on this pair a values-only Conv1d + GELU +
    # masked mean pooling returns exactly the same features for both patches.
    values, first, second = gap_pair()
    conv = torch.nn.Conv1d(1, 6, kernel_size=3, padding=1)

    def values_only(x, valid):
        weights = valid.to(x.dtype)
        features = torch.nn.functional.gelu(conv(torch.where(valid, x, 0.0)))
        return (features * weights).sum(-1) / weights.sum(-1).clamp_min(1)

    # Equal to float32 summation rounding: the two are the same feature vector.
    torch.testing.assert_close(values_only(values, first), values_only(values, second),
                               rtol=0, atol=1e-6)


def test_mask_channel_makes_missing_positions_identifiable():
    values, first, second = gap_pair()
    feature = WaveFeature(6).eval()
    with torch.no_grad():
        a, b = feature(values, first), feature(values, second)
    assert (a - b).abs().max() > 1e-3


def test_encoder_separates_windows_that_differ_only_in_where_readings_are_missing():
    encoder = GlucoFMEncoder(ModelConfig(dropout=0)).eval()
    glucose = torch.full((2, 288), 110.0)
    observed = torch.ones(2, 288, dtype=torch.bool)
    observed[0, 30] = False
    observed[1, 90] = False
    start = torch.zeros(2, dtype=torch.long)
    with torch.no_grad():
        out = encoder(glucose, observed, start)["embedding"]
    assert not torch.allclose(out[0], out[1])
