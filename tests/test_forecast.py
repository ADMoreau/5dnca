"""The forecast experiment: the energy score, exact targets, the observation clamp and
which candidate is which."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bounce  # noqa: E402
from train_forecast import MODES, Bounces, Forecast, best_exact, energy_score  # noqa: E402

CFG = bounce.BounceConfig()
ARGS = SimpleNamespace(candidates=3, two_way=False, channels=8, hidden=16, perception="sobel", no_compile=True)


@pytest.fixture(scope="module")
def book():
    return Bounces(CFG, 0.2, 0.95, (0.55, 0.70))


def test_energy_score_by_hand():
    c = torch.stack([torch.zeros(5), torch.ones(5)])[None]             # two candidates, 0 and 1
    es, d_y, d_cc = energy_score(c, torch.full((1, 5), 0.25), eps=0.0)
    assert torch.allclose(d_y, torch.tensor([[0.25, 0.75]]))
    assert torch.allclose(d_cc, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    assert torch.allclose(es, torch.tensor([0.5 - 2 * 1.0 / 8]))


def test_energy_score_rewards_candidates_that_cover_both_outcomes():
    a, b = torch.zeros(6), torch.ones(6)                                 # two equally likely outcomes

    def expected(c1, c2):
        c = torch.stack([c1, c2])[None].expand(2, -1, -1)
        return energy_score(c, torch.stack([a, b]), eps=0.0)[0].mean()

    both = expected(a, b)
    assert both < expected((a + b) / 2, (a + b) / 2)                    # hedging in the middle
    assert both < expected(a, a)                                        # covering only one outcome


def test_energy_score_pulls_candidates_together_when_the_outcome_is_certain():
    a = torch.zeros(6)
    tight = energy_score(torch.stack([a, a])[None], a[None], eps=0.0)[0]
    apart = energy_score(torch.stack([a - 0.5, a + 0.5])[None], a[None], eps=0.0)[0]
    assert tight < apart


def test_energy_score_gradient_is_finite_when_candidates_coincide():
    c = torch.zeros(1, 3, 10, requires_grad=True)
    energy_score(c, torch.ones(1, 10))[0].sum().backward()
    assert torch.isfinite(c.grad).all()


def test_best_exact_finds_the_quartiles_of_a_uniform_line():
    p = np.linspace(0, 10, 101)                                          # outcomes uniform on [0, 10]
    dist = np.abs(p[:, None] - p[None])
    every = np.arange(len(p))
    atoms, _ = best_exact(dist, every, every, 2)
    assert abs(p[atoms[0]] - 2.5) <= 0.2 and abs(p[atoms[1]] - 7.5) <= 0.2


def test_targets_are_exact_bounces(book):
    for e in (0.2, 0.37, 0.95):
        assert np.allclose(book.hist[book.index(e)].numpy(), bounce.render_history(e, CFG))
    trained, held = book.e[book.train], book.e[book.held]
    assert trained.min() == pytest.approx(0.2) and trained.max() == pytest.approx(0.95)
    assert not ((trained >= 0.55) & (trained <= 0.70)).any()
    assert held.min() == pytest.approx(0.55) and held.max() == pytest.approx(0.70)


@pytest.mark.parametrize("mode", MODES)
def test_observation_is_clamped_and_shared_by_every_candidate(book, mode):
    fc = Forecast(mode, CFG, ARGS)
    truth = book.hist[[book.index(0.3), book.index(0.9)]]
    n_obs = torch.tensor([0, 2])
    obs = fc.observation(truth, n_obs)
    x = fc.start(obs, 2)
    seen = [fc.fork + int(n) for n in n_obs]
    c = fc.cands(x)
    assert c.shape[:3] == (2, ARGS.candidates, ARGS.channels)
    for b in range(2):
        assert torch.equal(c[b, :, :, seen[b]:], torch.zeros_like(c[b, :, :, seen[b]:]))   # empty future
    with torch.no_grad():
        for p in fc.model.parameters():
            p.normal_(0, 0.1)
        torch.manual_seed(0)
        x = fc.rollout(x, 3, obs, fc.cond(2))
    c = fc.cands(x)
    for b in range(2):
        for k in range(ARGS.candidates):
            assert torch.equal(c[b, k, :4, :seen[b]], truth[b, :, :seen[b]])


@pytest.mark.parametrize("mode", MODES)
def test_each_candidate_reads_its_own_index(mode):
    fc = Forecast(mode, CFG, ARGS)
    B, K = 2, ARGS.candidates
    x = torch.zeros(fc.rows(B), ARGS.channels, *fc.grid)
    seen_by_cells = fc.cands(fc.cond(B).expand(x.shape[0], K, *x.shape[2:]))   # (B, K, K, W, Z, Y, X)
    for b in range(B):
        for k in range(K):
            v = seen_by_cells[b, k, :, fc.fork:].flatten(1)                     # the candidate's future
            assert torch.equal(v.amin(1), torch.eye(K)[k]) and torch.equal(v.amax(1), torch.eye(K)[k])
            if mode == "tree":                                                  # the shared trunk has no label
                assert not seen_by_cells[b, k, :, :fc.fork].any()


def _tree_state(fc, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, ARGS.channels, *fc.grid, generator=g)
    x[:, 3] = (torch.rand(1, *fc.grid, generator=g) > 0.3).float()            # mostly alive
    return x


@pytest.mark.parametrize("two_way", [False, True])
def test_tree_candidates_meet_only_through_the_fork(two_way):
    fc = Forecast("tree", CFG, SimpleNamespace(**(vars(ARGS) | {"two_way": two_way})))
    with torch.no_grad():
        for p in fc.model.parameters():
            p.normal_(0, 0.1)
        x = _tree_state(fc)
        assert fc.model.n_nodes == fc.fork + ARGS.candidates * (CFG.W - fc.fork)
        c = fc.cands(x)
        for k in range(1, ARGS.candidates):                                     # one trunk for everyone
            assert torch.equal(c[:, k, :, :fc.fork], c[:, 0, :, :fc.fork])
        x2 = x.clone()
        x2[:, :, fc._paths[0, fc.fork:]] = _tree_state(fc, seed=1)[:, :, fc._paths[0, fc.fork:]]   # change branch 0
        cond = fc.cond(1)
        a, b = x, x2
        for _ in range(4):
            torch.manual_seed(0)
            a = fc.model(a, cond=cond)
            torch.manual_seed(0)
            b = fc.model(b, cond=cond)
        others = fc._paths[1:].flatten()
        changed = not torch.equal(a[:, :, others], b[:, :, others])
        assert changed == two_way                                               # two-way: felt through the fork
