"""The tree NCA: a chain is exactly a 4D NCA, identical branches act like one line,
and one-way edges keep children from influencing parents."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nca  # noqa: E402
from tree_nca import TreeNCA  # noqa: E402

torch.manual_seed(0)
C, S = 8, (4, 5, 3)


def _random_state(n, batch=2):
    x = torch.rand(batch, C, n, *S)
    x[:, nca.ALPHA] = (torch.rand(batch, n, *S) > 0.6).float() * torch.rand(batch, n, *S)
    return x


def _fork_tree(trunk, branches, length):
    parent = list(range(-1, trunk - 1))
    for _ in range(branches):
        prev = trunk - 1
        for _ in range(length):
            parent.append(prev)
            prev = len(parent) - 1
    return parent


def _randomise(model):
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.1)
    return model


@pytest.mark.parametrize("perception", ["sobel", "isotropic"])
@pytest.mark.parametrize("padding", ["zeros", "circular"])
def test_chain_matches_4d_nca(perception, padding):
    n = 6
    ref = _randomise(nca.NCA(4, C, 32, perception=perception, padding=padding))
    tree = TreeNCA(list(range(-1, n - 1)), channels=C, hidden=32, perception=perception, padding=padding)
    tree.load_state_dict(ref.state_dict())
    x = _random_state(n)
    if padding == "circular":
        # nca.NCA wraps along w too; the tree never does. Compare away from the w ends.
        pr, pt = ref.perceive_cells(x), tree.perceive_cells(x)
        assert torch.allclose(pr[:, :, 1:-1], pt[:, :, 1:-1], atol=1e-5)
        return
    assert torch.allclose(ref.perceive_cells(x), tree.perceive_cells(x), atol=1e-5)
    assert torch.equal(ref.alive(x), tree.alive(x))
    torch.manual_seed(1)
    a = ref(x)
    torch.manual_seed(1)
    b = tree(x)
    assert torch.allclose(a, b, atol=1e-5)


def test_identical_branches_look_like_one_line():
    trunk, k, length = 3, 3, 4
    parent = _fork_tree(trunk, k, length)
    line = TreeNCA(list(range(-1, trunk + length - 1)), channels=C, hidden=32)
    tree = TreeNCA(parent, channels=C, hidden=32)
    xl = _random_state(trunk + length)
    xt = torch.cat([xl[:, :, :trunk]] + [xl[:, :, trunk:]] * k, dim=2)
    pl, pt = line.perceive_cells(xl), tree.perceive_cells(xt)
    al, at = line.alive(xl), tree.alive(xt)
    for b in range(k):
        path = list(range(trunk)) + list(range(trunk + b * length, trunk + (b + 1) * length))
        assert torch.allclose(pt[:, :, path], pl, atol=1e-6)
        assert torch.equal(at[:, :, path], al)


@pytest.mark.parametrize("edges,expect_change", [("down", False), ("both", True)])
def test_one_way_edges_hide_children_from_parents(edges, expect_change):
    trunk, k, length = 3, 2, 3
    parent = _fork_tree(trunk, k, length)
    model = TreeNCA(parent, edges=edges, channels=C, hidden=32)
    x = _random_state(len(parent))
    x2 = x.clone()
    x2[:, :, trunk:] = torch.rand_like(x2[:, :, trunk:])  # change every branch node
    p1, p2 = model.perceive_cells(x), model.perceive_cells(x2)
    a1, a2 = model.alive(x), model.alive(x2)
    trunk_same = torch.equal(p1[:, :, :trunk], p2[:, :, :trunk]) and torch.equal(a1[:, :, :trunk], a2[:, :, :trunk])
    assert trunk_same != expect_change


def test_fork_reads_mean_of_children():
    parent = [-1, 0, 1, 1]  # node 1 has two children
    model = TreeNCA(parent, channels=C, hidden=32)
    x = _random_state(4)
    xp = torch.nn.functional.pad(x, (1,) * 6)
    assert torch.allclose(model.down(xp)[:, :, 1], (xp[:, :, 2] + xp[:, :, 3]) / 2)
    assert torch.equal(model.down(xp)[:, :, 2], torch.zeros_like(xp[:, :, 2]))   # leaf
    assert torch.equal(model.up(xp)[:, :, 0], torch.zeros_like(xp[:, :, 0]))     # root


def test_cond_reaches_the_network_and_gradients_flow():
    parent = _fork_tree(2, 2, 2)
    model = _randomise(TreeNCA(parent, channels=C, hidden=32, cond_channels=2, skip_dead=True))
    x = _random_state(len(parent), batch=1)
    cond = torch.zeros(1, 2, len(parent), 1, 1, 1)
    cond2 = cond.clone()
    cond2[:, 1] = 1.0
    torch.manual_seed(3)
    a = model(x, cond=cond)
    torch.manual_seed(3)
    b = model(x, cond=cond2)
    assert not torch.allclose(a, b)
    model.rollout(x, 3, checkpoint=True, cond=cond2).square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
