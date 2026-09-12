"""Focused unit tests for the dependency-light contrastive loss module."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


@pytest.fixture()
def loss_module(monkeypatch):
    """Load the module without importing Pointcept's optional model backends."""

    class Registry:

        def register_module(self):
            return lambda cls: cls

    root = types.ModuleType("pointcept")
    models = types.ModuleType("pointcept.models")
    losses = types.ModuleType("pointcept.models.losses")
    builder = types.ModuleType("pointcept.models.losses.builder")
    builder.LOSSES = Registry()
    root.__path__, models.__path__, losses.__path__ = [], [], []
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "pointcept", root)
        context.setitem(sys.modules, "pointcept.models", models)
        context.setitem(sys.modules, "pointcept.models.losses", losses)
        context.setitem(sys.modules, "pointcept.models.losses.builder",
                        builder)
        name = "pointcept.models.losses._isolated_instance_contrastive"
        path = Path(
            __file__).parents[1] / "pointcept/models/losses/contrastive.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        context.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        yield module


def _rows(mined):
    selected = mined.selected_indices
    return {
        int(selected[a]):
        (selected[p[pm]].tolist(), selected[n[nm]].tolist(), int(c), int(k))
        for a, p, n, pm, nm, c, k in zip(
            mined.anchor_indices,
            mined.positive_indices,
            mined.negative_indices,
            mined.positive_mask,
            mined.negative_mask,
            mined.anchor_cloud,
            mined.anchor_class,
        )
    }


def test_mining_selects_farthest_positives_and_nearest_negatives(loss_module):
    loss = loss_module.InstanceContrastiveLoss(anchors_per_class=3,
                                               positives_per_anchor=2,
                                               negatives_per_anchor=2,
                                               num_classes=3)
    coord = torch.tensor([[0., 0, 0], [2, 0, 0], [9, 0, 0], [1, 0, 0],
                          [8, 0, 0]])
    target = torch.tensor([1, 1, 1, 2, 2])
    torch.manual_seed(3)
    rows = _rows(loss.mine(coord, target, torch.tensor([5])))
    assert rows[0][:2] == ([2, 1], [3, 4])
    assert rows[2][:2] == ([0, 1], [4, 3])


def test_mining_excludes_self_background_ignore_and_invalid_labels(
        loss_module):
    coord = torch.arange(7, dtype=torch.float32)[:, None].repeat(1, 3)
    target = torch.tensor([1, 1, 0, -1, 3, 99, -5])
    plain = loss_module.InstanceContrastiveLoss(anchors_per_class=2,
                                                positives_per_anchor=2,
                                                negatives_per_anchor=3,
                                                num_classes=3)
    assert plain.mine(coord, target,
                      torch.tensor([7])).anchor_indices.numel() == 0
    with_bg = loss_module.InstanceContrastiveLoss(
        anchors_per_class=2,
        positives_per_anchor=2,
        negatives_per_anchor=3,
        num_classes=3,
        include_background_negatives=True)
    rows = _rows(with_bg.mine(coord, target, torch.tensor([7])))
    assert set(rows) == {0, 1}
    for anchor, (positive, negative, _, _) in rows.items():
        assert anchor not in positive
        assert set(positive) == ({1} if anchor == 0 else {0})
        assert negative == [2]


def test_pairs_never_cross_cumulative_offset_cloud_boundaries(loss_module):
    loss = loss_module.InstanceContrastiveLoss(anchors_per_class=2,
                                               positives_per_anchor=1,
                                               negatives_per_anchor=1,
                                               num_classes=3)
    # Each cloud independently contains labels 1 and 2.  Cross-cloud points are closer.
    coord = torch.tensor([[0., 0, 0], [5, 0, 0], [1, 0, 0], [0.01, 0, 0],
                          [6, 0, 0], [1.01, 0, 0]])
    target = torch.tensor([1, 1, 2, 1, 1, 2])
    rows = _rows(loss.mine(coord, target, torch.tensor([3, 6])))
    assert rows
    for anchor, (positive, negative, cloud, _) in rows.items():
        lo, hi = ((0, 3) if cloud == 0 else (3, 6))
        assert lo <= anchor < hi
        assert all(lo <= x < hi for x in positive + negative)


def test_streamed_mining_matches_dense_reference_and_variable_budgets(
        loss_module):
    torch.manual_seed(8)
    coord = torch.randn(13, 3)
    target = torch.tensor([1] * 5 + [2] * 3 + [0, -1, 25, 2, 1])
    common = dict(anchors_per_class=10,
                  positives_per_anchor=4,
                  negatives_per_anchor=5,
                  num_classes=3)
    torch.manual_seed(21)
    streamed = loss_module.InstanceContrastiveLoss(candidate_chunk_size=2,
                                                   **common).mine(
                                                       coord, target,
                                                       torch.tensor([13]))
    torch.manual_seed(21)
    dense = loss_module.InstanceContrastiveLoss(candidate_chunk_size=100,
                                                **common).mine(
                                                    coord, target,
                                                    torch.tensor([13]))
    assert _rows(streamed) == _rows(dense)
    for positive, negative, _, _ in _rows(streamed).values():
        assert 1 <= len(positive) <= 4
        assert 1 <= len(negative) <= 5


def test_unequal_small_clouds_keep_fixed_pair_widths_and_safe_padding(
        loss_module):
    loss = loss_module.InstanceContrastiveLoss(anchors_per_class=4,
                                               positives_per_anchor=4,
                                               negatives_per_anchor=5,
                                               num_classes=3)
    coord = torch.arange(7, dtype=torch.float32)[:, None].repeat(1, 3)
    # The first cloud has fewer candidates than either budget; the second does not.
    mined = loss.mine(coord, torch.tensor([1, 1, 2, 1, 1, 1, 2]),
                      torch.tensor([3, 7]))
    assert mined.positive_indices.shape[1] == 4
    assert mined.negative_indices.shape[1] == 5
    assert mined.positive_indices[~mined.positive_mask].eq(0).all()
    assert mined.negative_indices[~mined.negative_mask].eq(0).all()
    assert mined.positive_indices.max() < mined.selected_indices.numel()
    assert mined.negative_indices.max() < mined.selected_indices.numel()


def test_invalid_trailing_candidates_never_remap_or_select(loss_module):
    loss = loss_module.InstanceContrastiveLoss(anchors_per_class=2,
                                               positives_per_anchor=2,
                                               negatives_per_anchor=2,
                                               num_classes=3)
    coord = torch.arange(5, dtype=torch.float32)[:, None].repeat(1, 3)
    mined = loss.mine(coord, torch.tensor([1, 1, 2, -1, 0]), torch.tensor([5]))
    assert set(mined.selected_indices.tolist()) == {0, 1, 2}
    # Invalid padded entries must not point one past the compact embedding.
    assert mined.positive_indices.max() < mined.selected_indices.numel()
    assert mined.negative_indices.max() < mined.selected_indices.numel()


def test_configurable_ignore_excludes_foreground_and_background_candidates(
        loss_module):
    coord = torch.arange(4, dtype=torch.float32)[:, None].repeat(1, 3)
    ignored_rib = loss_module.InstanceContrastiveLoss(anchors_per_class=2,
                                                      positives_per_anchor=1,
                                                      negatives_per_anchor=2,
                                                      num_classes=4,
                                                      ignore_index=2)
    rows = _rows(
        ignored_rib.mine(coord, torch.tensor([1, 1, 2, 3]), torch.tensor([4])))
    assert rows and all(2 not in negative
                        for _, negative, _, _ in rows.values())
    ignored_background = loss_module.InstanceContrastiveLoss(
        anchors_per_class=2,
        positives_per_anchor=1,
        negatives_per_anchor=2,
        num_classes=3,
        ignore_index=0,
        include_background_negatives=True)
    assert ignored_background.mine(coord[:3], torch.tensor([1, 1, 0]),
                                   torch.tensor(
                                       [3])).anchor_indices.numel() == 0


def test_validates_integral_inputs_finite_coordinates_and_empty_cloud_endpoints(
        loss_module):
    with pytest.raises(ValueError, match="integral"):
        loss_module.InstanceContrastiveLoss().mine(torch.zeros(1, 3),
                                                   torch.tensor([0]),
                                                   torch.tensor([1.]))
    with pytest.raises(ValueError, match="finite"):
        loss_module.InstanceContrastiveLoss().mine(
            torch.tensor([[float("nan"), 0., 0.]]), torch.tensor([0]),
            torch.tensor([1]))
    with pytest.raises(ValueError, match="finite"):
        loss_module.InstanceContrastiveLoss(temperature=float("inf"))
    with pytest.raises(ValueError, match="background"):
        loss_module.InstanceContrastiveLoss(bg_class=25, num_classes=25)
    loss = loss_module.InstanceContrastiveLoss(anchors_per_class=2,
                                               positives_per_anchor=1,
                                               negatives_per_anchor=1,
                                               num_classes=3)
    mined = loss.mine(
        torch.arange(3, dtype=torch.float32)[:, None].repeat(1, 3),
        torch.tensor([1, 1, 2]), torch.tensor([0, 3]))
    assert mined.num_clouds == 2 and mined.anchor_cloud.eq(1).all()


def test_empty_pairs_return_graph_connected_zero(loss_module):
    loss = loss_module.InstanceContrastiveLoss(num_classes=3)
    mined = loss.mine(torch.zeros(1, 3), torch.tensor([0]), torch.tensor([1]))
    projection = torch.nn.Linear(4, 3)
    value = loss(projection(torch.randn(1, 4)), mined)
    value.backward()
    assert value.item() == 0
    assert all(parameter.grad is not None
               for parameter in projection.parameters())


def _mined(module, clouds, classes, positive_count=1, negative_count=1):
    count = len(clouds)
    # one orthogonal-ish anchor / positive / negative triple per row
    selected = torch.arange(count * 3)
    anchor = torch.arange(count) * 3
    positive = (anchor + 1).unsqueeze(1).expand(-1, positive_count).clone()
    negative = (anchor + 2).unsqueeze(1).expand(-1, negative_count).clone()
    return module.MinedPairs(selected, anchor, positive, negative,
                             torch.ones_like(positive, dtype=torch.bool),
                             torch.ones_like(negative, dtype=torch.bool),
                             torch.tensor(clouds), torch.tensor(classes), 3,
                             count)


def test_hierarchical_reduction_weights_ribs_equally_and_includes_empty_cloud(
        loss_module):
    loss = loss_module.InstanceContrastiveLoss(loss_weight=1, temperature=1)
    # Cloud 0: rib 1 has two anchors of loss log(1 + exp(-2)); rib 2 has one log(2).
    # Cloud 1 has no anchors, cloud 2 has rib 1 with log(2).  Mean is over three clouds.
    mined = _mined(loss_module, [0, 0, 0, 2], [1, 1, 2, 1])
    emb = torch.tensor([[1., 0.], [1., 0.], [-1., 0.], [1., 0.], [1., 0.],
                        [-1., 0.], [1., 0.], [0., 1.], [0., 1.], [1., 0.],
                        [0., 1.], [0., 1.]])
    got = loss(emb, mined)
    low = torch.log1p(torch.exp(torch.tensor(-2.)))
    expected = ((low + torch.log(torch.tensor(2.))) / 2 + 0 +
                torch.log(torch.tensor(2.))) / 3
    assert torch.allclose(got, expected)


def test_each_positive_has_its_own_denominator_and_equal_logits_are_log17(
        loss_module):
    loss = loss_module.InstanceContrastiveLoss(loss_weight=1, temperature=1)
    mined = loss_module.MinedPairs(torch.arange(19), torch.tensor([0]),
                                   torch.tensor([[1, 2]]),
                                   torch.arange(3, 19).view(1, 16),
                                   torch.tensor([[True, True]]),
                                   torch.ones((1, 16), dtype=torch.bool),
                                   torch.tensor([0]), torch.tensor([1]), 1, 18)
    # p0 is aligned and p1 anti-aligned: p1 must not enter p0's denominator.
    emb = torch.tensor([[1., 0.], [1., 0.], [-1., 0.]] + [[1., 0.]] * 16)
    expected = (torch.log(torch.tensor(17.)) +
                torch.log1p(16 * torch.exp(torch.tensor(2.)))) / 2
    assert torch.allclose(loss(emb, mined), expected)
    equal = torch.ones(18, 2)
    equal_mined = loss_module.MinedPairs(torch.arange(18), torch.tensor([0]),
                                         torch.tensor([[1]]),
                                         torch.arange(2, 18).view(1, 16),
                                         torch.tensor([[True]]),
                                         torch.ones((1, 16), dtype=torch.bool),
                                         torch.tensor([0]), torch.tensor([1]),
                                         1, 18)
    assert torch.allclose(loss(equal, equal_mined),
                          torch.log(torch.tensor(17.)))


def test_fp32_loss_under_autocast_and_zero_weight_preserves_gradients(
        loss_module):
    mined = _mined(loss_module, [0], [1])
    embedding = torch.tensor([[1., 0.], [0., 1.], [1., 0.]],
                             requires_grad=True)
    weighted = loss_module.InstanceContrastiveLoss(loss_weight=0,
                                                   temperature=.1)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        value = weighted(embedding.to(torch.bfloat16), mined)
    assert value.dtype == torch.float32 and torch.isfinite(value)
    value.backward()
    assert embedding.grad is not None
