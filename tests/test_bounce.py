"""Physical and structural checks on the exact bouncing-ball targets."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bounce  # noqa: E402

CFG = bounce.BounceConfig()


def test_first_impact_lands_at_configured_w():
    t_hit = np.sqrt(2 * (CFG.drop_height - CFG.radius) / CFG.gravity)
    assert t_hit == pytest.approx(CFG.impact_w)
    assert bounce.height(t_hit, 0.5, CFG) == pytest.approx(CFG.radius)


@pytest.mark.parametrize("e", CFG.bounciness)
def test_ball_never_goes_through_the_floor(e):
    hs = [bounce.height(t, e, CFG) for t in np.linspace(0, 60, 3001)]
    assert min(hs) >= CFG.radius - 1e-9


@pytest.mark.parametrize("e", CFG.bounciness)
def test_rebound_peaks_follow_restitution(e):
    # Peak height above the resting height scales by e^2 per bounce.
    g, t_hit = CFG.gravity, CFG.impact_w
    v1 = e * g * t_hit
    peak1 = CFG.radius + v1**2 / (2 * g)
    ts = np.linspace(t_hit, t_hit + 2 * v1 / g, 2001)
    assert max(bounce.height(t, e, CFG) for t in ts) == pytest.approx(peak1, rel=1e-4)
    first_drop = CFG.drop_height - CFG.radius
    assert (peak1 - CFG.radius) / first_drop == pytest.approx(e**2)


def test_histories_identical_before_impact_and_different_after():
    lat = bounce.lattice_target(CFG)
    fork = bounce.first_impact_index(CFG)
    pre, post = lat[:, :, :fork], lat[:, :, fork:]
    assert all(np.array_equal(pre[:, 0], pre[:, v]) for v in range(1, lat.shape[1]))
    for v in range(1, lat.shape[1]):
        assert not np.array_equal(post[:, 0], post[:, v])


def test_ball_mass_is_constant_and_inside_the_grid():
    # The whole ball stays inside the box, so total alpha is the same in every slice.
    lat = bounce.lattice_target(CFG)
    mass = lat[3].sum(axis=(2, 3, 4))
    assert np.allclose(mass, mass[0, 0], rtol=0.02)


def test_tree_matches_lattice_along_every_branch():
    lat = bounce.lattice_target(CFG)
    tree = bounce.tree_target(CFG, lat)
    parent, node_w = tree["parent"], tree["node_w"]
    leaves = [i for i in range(len(parent)) if i not in set(parent.tolist())]
    assert len(leaves) == len(CFG.bounciness)
    for b, leaf in enumerate(leaves):
        path, n = [], leaf
        while n >= 0:
            path.append(n)
            n = parent[n]
        path = path[::-1]
        assert [node_w[i] for i in path] == list(range(CFG.W))
        assert np.array_equal(tree["nodes"][path], np.moveaxis(lat[:, b], 1, 0))


def test_tree_shape_and_edges():
    tree = bounce.tree_target(CFG)
    fork, K = tree["fork_w"], len(CFG.bounciness)
    assert len(tree["parent"]) == fork + K * (CFG.W - fork)
    edges = bounce.tree_edges(tree["parent"])
    assert len(edges) == len(tree["parent"]) - 1           # a tree: one edge per non-root node
    children_of_last_trunk = (edges[:, 0] == fork - 1).sum()
    assert children_of_last_trunk == K                       # the fork
