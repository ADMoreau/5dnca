"""The hierarchical NCA: pooling and upsampling against PyTorch, the flat rule at one level
and before training, how far one step reaches, and one-way time."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nca  # noqa: E402
from hier_nca import HierNCA, pool2, upsample2  # noqa: E402

C = 8


def _randomise(model, seed=0, std=0.1):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * std)
    return model


def _state(shape, batch=2, seed=0, alive=None):
    """Random channels; alpha mostly alive and partly dead, or 1 everywhere with alive=True."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, C, *shape, generator=g, dtype=torch.float64)
    if alive:
        x[:, nca.ALPHA] = 1.0
    else:
        x[:, nca.ALPHA] = (torch.rand(batch, *shape, generator=g, dtype=torch.float64) > 0.4).double()
    return x


# ---------------------------------------------------------------- pooling and upsampling

@pytest.mark.parametrize("level", [1, 2])
def test_pooling_matches_torch(level):
    x = torch.randn(2, 3, 8, 12, 16, dtype=torch.float64)
    mean, top = x, x
    for _ in range(level):
        mean, top = pool2(mean, (0, 1, 2), "mean"), pool2(top, (0, 1, 2), "max")
    assert torch.allclose(mean, F.avg_pool3d(x, 2 ** level))
    assert torch.equal(top, F.max_pool3d(x, 2 ** level))


@pytest.mark.parametrize("shape", [(9, 12, 5, 7), (1, 4, 1, 5)])
@pytest.mark.parametrize("axes", [(0, 1, 2, 3), (1, 2, 3), (0, 2, 3), (0, 1)])
def test_max_pooling_matches_pairwise_maxima(shape, axes):
    """The max-pool kernel path (trailing axes) and the pairwise path must agree, odd lengths included."""
    x = torch.randn(2, 3, *shape, dtype=torch.float64)
    ref = x
    for a in axes:
        d, n = 2 + a, ref.shape[2 + a]
        if n == 1:
            continue
        if n % 2:
            ref = torch.cat([ref, ref.narrow(d, n - 1, 1)], dim=d)
        ref = ref.unflatten(d, (ref.shape[d] // 2, 2)).amax(d + 1)
    assert torch.equal(pool2(x, axes, "max"), ref)


def test_pooling_an_odd_length_counts_only_real_cells():
    x = torch.arange(5.0).view(1, 1, 5)
    assert torch.equal(pool2(x, (0,), "mean"), torch.tensor([[[0.5, 2.5, 4.0]]]))
    assert torch.equal(pool2(x, (0,), "max"), torch.tensor([[[1.0, 3.0, 4.0]]]))


def test_pooling_leaves_other_axes_alone():
    x = torch.randn(1, 2, 6, 8, 10)
    assert pool2(x, (1, 2), "mean").shape == (1, 2, 6, 4, 5)


def test_upsampling_is_linear_interpolation():
    c = torch.randn(2, 3, 4, 6, 8, dtype=torch.float64)
    ref = F.interpolate(c, scale_factor=2, mode="trilinear", align_corners=False)
    assert torch.allclose(upsample2(c, (0, 1, 2), (8, 12, 16)), ref)
    assert torch.allclose(upsample2(c, (0, 1, 2), (7, 12, 15)), ref[:, :, :7, :, :15])  # odd finer lengths


def test_circular_upsampling_wraps():
    c = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    up = upsample2(c, (0,), (8,), "circular")
    assert up[0, 0, 0] == 0.75 and up[0, 0, -1] == 0.25  # the last cell borrows from the first


# ---------------------------------------------------------------- the flat rule inside

def test_one_level_is_the_flat_nca():
    flat = _randomise(nca.NCA(3, C, 16, skip_dead=True)).double()
    hier = HierNCA(3, C, 16, levels=1, skip_dead=True).double()
    hier.load_state_dict(flat.state_dict())
    x = _state((6, 7, 5))
    torch.manual_seed(1)
    a = flat(x)
    torch.manual_seed(1)
    b = hier(x)
    assert torch.equal(a, b)


@pytest.mark.parametrize("skip_dead", [False, True])
def test_an_untrained_hierarchy_acts_like_the_flat_nca(skip_dead):
    """The coarse networks start at zero, so the context is zero until training changes them."""
    hier = HierNCA(4, C, 16, levels=3, skip_dead=skip_dead, shift_blocks=False).double()
    flat = nca.NCA(4, C, 16, skip_dead=skip_dead).double()
    with torch.no_grad():
        torch.nn.init.normal_(hier.fc1.weight, std=0.1)
        flat.fc1.weight.copy_(hier.fc1.weight)
        flat.fc0.weight.copy_(hier.fc0.weight[:, : flat.fc0.weight.shape[1]])
        flat.fc0.bias.copy_(hier.fc0.bias)
    x = _state((4, 6, 6, 6))
    assert not hier.context(x).any()
    torch.manual_seed(2)
    a = hier(x)
    torch.manual_seed(2)
    b = flat(x)
    assert torch.allclose(a, b)


def test_skip_dead_matches_dense_rule_with_masked_updates():
    model = _randomise(HierNCA(3, C, 16, levels=3, skip_dead=True, shift_blocks=False)).double()
    x = _state((6, 8, 6)).requires_grad_()
    torch.manual_seed(7)
    out = model(x)

    torch.manual_seed(7)
    pre = model.alive(x)
    y = torch.cat([nca.perceive(x), model.context(x)], dim=1)
    fire = torch.rand(2, 1, *x.shape[2:]) <= model.fire_rate
    b, p = y.shape[:2]
    dx = model.fc1(torch.relu(model.fc0(y.reshape(b, p, -1)))).reshape(x.shape)
    ref = x + dx * (fire & pre).double()
    ref = ref * (pre & model.alive(ref)).double()
    assert torch.allclose(out, ref)
    g_out, = torch.autograd.grad(out.sum(), x)
    g_ref, = torch.autograd.grad(ref.sum(), x)
    assert torch.allclose(g_out, g_ref)


def test_cond_reaches_the_fine_rule_alongside_the_context():
    model = _randomise(HierNCA(3, C, 16, levels=2, cond_channels=2, shift_blocks=False)).double()
    x = _state((4, 6, 6), alive=True)
    cond = torch.zeros(2, 2, 1, 1, 1, dtype=torch.float64)
    with torch.no_grad():
        a = model(x, fire_rate=1.0, cond=cond)
        b = model(x, fire_rate=1.0, cond=cond + 1.0)
    assert not torch.equal(a, b)


# ---------------------------------------------------------------- reach

def _changed_by(model, x, cell):
    """Cells whose next state changes when one cell's hidden channels change (all cells fire)."""
    x2 = x.clone()
    x2[(slice(None), slice(4, None)) + tuple(cell)] += 1.0
    with torch.no_grad():
        return (model(x, fire_rate=1.0) != model(x2, fire_rate=1.0)).any(1).any(0)


@pytest.mark.parametrize("levels", [1, 2, 3, 4, 5])
def test_one_step_reaches_further_with_every_level(levels):
    """In each direction a step reaches at least 2**levels - 1 cells and at most
    3 * 2**(levels - 1) - 2, depending on where the cell sits in its blocks."""
    model = _randomise(HierNCA(1, C, 16, levels=levels, shift_blocks=False)).double()
    x = _state((256,), batch=1, alive=True)
    lo, hi = 2 ** levels - 1, 3 * 2 ** (levels - 1) - 2
    reach = []
    for p in range(100, 132):
        idx = _changed_by(model, x, (p,)).nonzero().squeeze(1)
        left, right = p - int(idx.min()), int(idx.max()) - p
        assert lo <= left <= hi and lo <= right <= hi
        reach.append(max(left, right))
    assert max(reach) == hi


def test_enough_levels_make_a_step_global():
    x = _state((16, 16, 16), batch=1, alive=True)
    capped = _randomise(HierNCA(3, C, 16, levels=2, shift_blocks=False)).double()
    full = _randomise(HierNCA(3, C, 16, levels=5, shift_blocks=False)).double()  # 16 -> 8 -> 4 -> 2 -> 1
    assert not _changed_by(capped, x, (0, 0, 0))[15, 15, 15]
    assert _changed_by(full, x, (0, 0, 0))[15, 15, 15]


# ---------------------------------------------------------------- moving the blocks

@pytest.mark.parametrize("offsets", [(0, 0, 0), (1, 2, 3), (3, 0, 1)])
def test_moved_blocks_follow_a_moved_grid_exactly(offsets):
    """Circular padding: move the content by s and the blocks with it, and the context is the
    old one moved by s. So no position is special once the offsets are random."""
    model = _randomise(HierNCA(3, C, 16, levels=3, padding="circular")).double()
    x = _state((8, 12, 8))
    s = (1, 3, 2)
    moved = [(o - d) % model.block for o, d in zip(offsets, s)]
    a = model.shifted_context(torch.roll(x, s, (2, 3, 4)), moved)
    b = torch.roll(model.shifted_context(x, offsets), s, (2, 3, 4))
    assert torch.allclose(a, b)


def test_moved_blocks_follow_moved_content_with_zero_padding():
    """Zero padding: the same holds wherever the world's edges are out of reach."""
    model = _randomise(HierNCA(2, C, 16, levels=3)).double()
    x = torch.zeros(1, C, 48, 48, dtype=torch.float64)
    x[:, :, 20:28, 21:29] = _state((8, 8), batch=1)
    s, offsets = (3, 1), (1, 2)
    moved = [(o - d) % model.block for o, d in zip(offsets, s)]
    a = model.shifted_context(torch.roll(x, s, (2, 3)), moved)
    b = torch.roll(model.shifted_context(x, offsets), s, (2, 3))
    assert torch.allclose(a[..., 16:34, 16:34], b[..., 16:34, 16:34])
    assert a[..., 16:34, 16:34].abs().sum() > 0


def test_fixed_blocks_are_a_coordinate_grid():
    """Without moving the blocks, moving the content alone changes the context: what the
    first 4D ball model used to snap its ball to multiples of 4 cells."""
    model = _randomise(HierNCA(3, C, 16, levels=3, padding="circular", shift_blocks=False)).double()
    x = _state((8, 12, 8))
    s = (1, 3, 2)
    assert not torch.allclose(model.context(torch.roll(x, s, (2, 3, 4))), torch.roll(model.context(x), s, (2, 3, 4)))


def test_zero_offsets_change_nothing_with_circular_padding():
    model = _randomise(HierNCA(3, C, 16, levels=3, padding="circular")).double()
    x = _state((8, 12, 8))
    assert torch.equal(model.shifted_context(x, (0, 0, 0)), model.context(x))


# ---------------------------------------------------------------- one-way time

def test_pooling_skips_the_one_way_axis():
    assert HierNCA(4, C, 16, levels=2, causal_axis=0).pool_axes == (1, 2, 3)
    with pytest.raises(ValueError):
        HierNCA(4, C, 16, levels=2, causal_axis=0, pool_axes=(0, 1, 2, 3))


def test_later_frames_never_reach_earlier_ones():
    model = _randomise(HierNCA(4, C, 32, levels=3, causal_axis=0, skip_dead=True)).double()
    shape = (6, 8, 8, 8)
    x = _state(shape)
    x2 = x.clone()
    x2[:, :, 3:] = _state(shape, seed=1)[:, :, 3:]
    torch.manual_seed(2)
    a = model(x)
    torch.manual_seed(2)
    b = model(x2)
    assert torch.equal(a[:, :, :3], b[:, :, :3])
    assert not torch.equal(a[:, :, 3:], b[:, :, 3:])


def test_time_still_moves_one_frame_per_step():
    """Space gets the long reach; along the one-way axis a step still reaches only the next frame."""
    model = _randomise(HierNCA(4, C, 16, levels=3, causal_axis=0, shift_blocks=False)).double()
    x = _state((5, 8, 8, 8), batch=1, alive=True)
    changed = _changed_by(model, x, (0, 0, 0, 0))
    assert changed[1].any() and not changed[2:].any()
    assert changed[0, 7, 7, 7]  # but across the whole frame


# ---------------------------------------------------------------- training

def test_gradients_reach_every_level():
    model = _randomise(HierNCA(3, C, 16, levels=3)).float()
    x = _state((8, 8, 8)).float()
    model.rollout(x, 2)[:, :4].pow(2).sum().backward()
    for net in model.coarse:
        for prm in net.parameters():
            assert prm.grad is not None and prm.grad.abs().sum() > 0


def test_checkpointed_rollout_gives_same_gradients():
    model = _randomise(HierNCA(4, C, 16, levels=2))
    x0 = model.seed((5, 6, 6, 6), batch=2)
    grads = []
    for ckpt in (False, True):
        torch.manual_seed(123)
        model.zero_grad()
        model.rollout(x0, 4, checkpoint=ckpt)[:, :4].pow(2).sum().backward()
        grads.append([p.grad.clone() for p in model.parameters()])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-6)


def test_5d_lattice_layout():
    """(v, w, z, y, x) with one-way time on w: only z, y and x get coarser."""
    model = _randomise(HierNCA(5, C, 16, levels=3, causal_axis=1, pool_axes=(2, 3, 4)))
    x = model.seed((3, 4, 8, 12, 16), batch=1)
    assert model.rollout(x, 2).shape == x.shape
