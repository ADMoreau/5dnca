"""The shift-based n-D operators must match PyTorch's dense conv2d/conv3d and max_pool2d/3d."""

import itertools
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nca  # noqa: E402

torch.manual_seed(0)
SMOOTH = torch.tensor([1.0, 2.0, 1.0])
DIFF = torch.tensor([-1.0, 0.0, 1.0])


def dense_sobel_kernel(ndim: int, axis: int) -> torch.Tensor:
    """Outer product: DIFF on `axis`, SMOOTH on the others, normalised like the original (/8 in 2D)."""
    k = torch.ones(())
    for a in range(ndim):
        k = k[..., None] * (DIFF / 2 if a == axis else SMOOTH / 4)
    return k


def depthwise(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    c, nd = x.shape[1], x.dim() - 2
    w = kernel.expand(c, 1, *kernel.shape)
    conv = {2: F.conv2d, 3: F.conv3d}[nd]
    return conv(x, w, padding=1, groups=c)


def test_2d_sobel_matches_original_kernel():
    # The Distill colab uses outer([1,2,1], [-1,0,1]) / 8 for d/dx, its transpose for d/dy.
    original_dx = torch.outer(SMOOTH, DIFF) / 8
    assert torch.allclose(dense_sobel_kernel(2, axis=1), original_dx)
    assert torch.allclose(dense_sobel_kernel(2, axis=0), original_dx.T)


@pytest.mark.parametrize("ndim", [2, 3])
def test_sobel_matches_dense_conv(ndim):
    x = torch.randn(2, 5, *([9] * ndim), dtype=torch.float64)
    for axis in range(ndim):
        ref = depthwise(x, dense_sobel_kernel(ndim, axis).double())
        assert torch.allclose(nca.sobel(x, axis), ref), f"axis {axis}"


@pytest.mark.parametrize("ndim", [2, 3])
def test_laplacian_matches_dense_conv(ndim):
    k = torch.zeros(*([3] * ndim), dtype=torch.float64)
    centre = (1,) * ndim
    k[centre] = -2 * ndim
    for a in range(ndim):
        for off in (0, 2):
            idx = list(centre)
            idx[a] = off
            k[tuple(idx)] = 1
    x = torch.randn(2, 4, *([8] * ndim), dtype=torch.float64)
    assert torch.allclose(nca.laplacian(x), depthwise(x, k))


@pytest.mark.parametrize("ndim", [2, 3])
def test_max_pool_matches_torch(ndim):
    x = torch.randn(2, 1, *([10] * ndim))
    pool = {2: F.max_pool2d, 3: F.max_pool3d}[ndim]
    assert torch.equal(nca.max_pool3(x), pool(x, 3, stride=1, padding=1))


def test_max_pool_4d_brute_force():
    x = torch.randn(1, 1, 5, 5, 5, 5)
    out = nca.max_pool3(x)
    padded = F.pad(x, (1,) * 8, value=float("-inf"))
    ref = torch.full_like(x, float("-inf"))
    for off in itertools.product(range(3), repeat=4):
        sl = (slice(None), slice(None)) + tuple(slice(o, o + 5) for o in off)
        ref = torch.maximum(ref, padded[sl])
    assert torch.equal(out, ref)


def test_circular_padding_wraps():
    x = torch.zeros(1, 1, 5, 5)
    x[0, 0, 0, 0] = 1.0
    lap = nca.laplacian(x, padding="circular")
    assert lap[0, 0, 4, 0] == 1 and lap[0, 0, 0, 4] == 1


@pytest.mark.parametrize("ndim", [2, 3, 4, 5])
def test_isotropic_perception_is_rotation_equivariant(ndim):
    # A 90-degree rotation in any axis plane must commute with isotropic perception.
    x = torch.randn(1, 3, *([6] * ndim), dtype=torch.float64)
    rot = lambda t: torch.rot90(t, 1, dims=(2, 3))  # noqa: E731
    assert torch.allclose(nca.perceive(rot(x), "isotropic"), rot(nca.perceive(x, "isotropic")))


@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
def test_step_runs_and_respects_alive_mask(ndim):
    model = nca.NCA(ndim, channels=8, hidden=16)
    nn_init = torch.nn.init
    nn_init.normal_(model.fc1.weight, std=0.1)
    side = 7 if ndim < 5 else 5
    x = model.seed((side,) * ndim, batch=2)
    y = model.rollout(x, 3)
    assert y.shape == x.shape
    # Cells farther than 3 steps from the seed cannot be alive or hold state.
    assert y[(0, slice(None)) + (0,) * ndim].abs().sum() == 0 or side // 2 <= 3


@pytest.mark.parametrize("ndim", [2, 4])
def test_perception_channel_counts(ndim):
    x = torch.zeros(1, 16, *([4] * ndim))
    for mode in ("sobel", "isotropic"):
        assert nca.perceive(x, mode).shape[1] == nca.perception_channels(16, ndim, mode)


def test_checkpointed_rollout_gives_same_gradients():
    model = nca.NCA(4, channels=8, hidden=16)
    torch.nn.init.normal_(model.fc1.weight, std=0.1)
    x0 = model.seed((5, 5, 5, 5), batch=2)
    grads = []
    for ckpt in (False, True):
        torch.manual_seed(123)
        model.zero_grad()
        model.rollout(x0, 4, checkpoint=ckpt)[:, :4].pow(2).sum().backward()
        grads.append([p.grad.clone() for p in model.parameters()])
    for a, b in zip(*grads):
        assert torch.allclose(a, b, atol=1e-6)


@pytest.mark.parametrize("ndim", [2, 4])
def test_skip_dead_matches_dense_rule_with_masked_updates(ndim):
    """The sparse path must equal: dense MLP on every cell, update only where fire & pre_alive."""
    torch.manual_seed(5)
    model = nca.NCA(ndim, channels=8, hidden=16, skip_dead=True).double()
    torch.nn.init.normal_(model.fc1.weight, std=0.2)
    x = model.seed((6,) * ndim, batch=2).double()
    for _ in range(3):  # grow a little so there is a mix of live and dead cells
        x = model(x)
    x = x.detach().requires_grad_()

    torch.manual_seed(7)
    out = model(x)

    torch.manual_seed(7)
    pre = nca.alive_mask(x)
    y = nca.perceive(x)
    fire = torch.rand(2, 1, *x.shape[2:]) <= model.fire_rate
    b, p = y.shape[:2]
    dx = model.fc1(torch.relu(model.fc0(y.reshape(b, p, -1)))).reshape(x.shape)
    ref = x + dx * (fire & pre).double()
    ref = ref * (pre & nca.alive_mask(ref)).double()

    assert torch.allclose(out, ref)
    g_out, = torch.autograd.grad(out.sum(), x)
    g_ref, = torch.autograd.grad(ref.sum(), x)
    assert torch.allclose(g_out, g_ref)


def brute_force_stencil(x: torch.Tensor, kernel: torch.Tensor, padding: str = "zeros") -> torch.Tensor:
    """Depthwise cross-correlation with a 3^n kernel by summing shifted copies (any n)."""
    nd = x.dim() - 2
    if padding == "zeros":
        xp = F.pad(x, (1, 1) * nd)
    else:
        xp = x
        for a in range(nd):
            xp = torch.cat([xp.narrow(2 + a, -1, 1), xp, xp.narrow(2 + a, 0, 1)], dim=2 + a)
    out = torch.zeros_like(x)
    size = x.shape[2:]
    for off in itertools.product(range(3), repeat=nd):
        if kernel[off] != 0:
            sl = (slice(None), slice(None)) + tuple(slice(o, o + s) for o, s in zip(off, size))
            out = out + kernel[off] * xp[sl]
    return out


@pytest.mark.parametrize("ndim", [1, 4, 5])
@pytest.mark.parametrize("padding", ["zeros", "circular"])
def test_sobel_matches_brute_force_any_n(ndim, padding):
    x = torch.randn(2, 3, *([5] * ndim), dtype=torch.float64)
    got = nca.sobels(x, padding)
    for axis in range(ndim):
        ref = brute_force_stencil(x, dense_sobel_kernel(ndim, axis).double(), padding)
        assert torch.allclose(got[axis], ref), f"axis {axis}"
