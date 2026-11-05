"""One-way time in nca.NCA: on a chain it is the one-way tree, and later frames never
reach earlier ones."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nca  # noqa: E402
from tree_nca import TreeNCA  # noqa: E402

C = 8


def _state(shape, batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(batch, C, *shape, generator=g)
    x[:, nca.ALPHA] = (torch.rand(batch, *shape, generator=g) > 0.6).float() * torch.rand(batch, *shape, generator=g)
    return x


def _randomise(model, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return model


@pytest.mark.parametrize("perception", ["sobel", "isotropic"])
@pytest.mark.parametrize("padding", ["zeros", "circular"])
def test_causal_chain_is_the_one_way_tree(perception, padding):
    n = 6
    ref = _randomise(nca.NCA(4, C, 32, perception=perception, padding=padding, causal_axis=0))
    tree = TreeNCA(list(range(-1, n - 1)), edges="down", channels=C, hidden=32,
                   perception=perception, padding=padding)
    tree.load_state_dict(ref.state_dict())
    x = _state((n, 4, 5, 3))
    assert torch.allclose(ref.perceive_cells(x), tree.perceive_cells(x), atol=1e-5)
    assert torch.equal(ref.alive(x), tree.alive(x))
    torch.manual_seed(1)
    a = ref(x)
    torch.manual_seed(1)
    b = tree(x)
    assert torch.allclose(a, b, atol=1e-5)


@pytest.mark.parametrize("ndim,axis", [(4, 0), (5, 1)])
@pytest.mark.parametrize("perception", ["sobel", "isotropic"])
def test_later_frames_never_reach_earlier_ones(ndim, axis, perception):
    shape = (6, 4, 5, 3) if ndim == 4 else (3, 6, 4, 3, 4)
    k = 3
    later = [slice(None)] * (2 + ndim)
    later[2 + axis] = slice(k, None)
    early = [slice(None)] * (2 + ndim)
    early[2 + axis] = slice(0, k)
    later, early = tuple(later), tuple(early)
    x = _state(shape)
    x2 = x.clone()
    x2[later] = _state(shape, seed=1)[later]
    for causal in (axis, None):
        model = _randomise(nca.NCA(ndim, C, 32, perception=perception, skip_dead=True, causal_axis=causal))
        torch.manual_seed(2)
        a = model(x)
        torch.manual_seed(2)
        b = model(x2)
        unchanged = (torch.equal(model.perceive_cells(x)[early], model.perceive_cells(x2)[early])
                     and torch.equal(model.alive(x)[early], model.alive(x2)[early])
                     and torch.equal(a[early], b[early]))
        assert unchanged == (causal is not None)


def test_causal_axis_must_be_an_axis():
    with pytest.raises(ValueError):
        nca.NCA(4, C, 32, causal_axis=4)
