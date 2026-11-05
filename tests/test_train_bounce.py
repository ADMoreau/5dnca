"""The four training setups see the same histories, conditions and seed position."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bounce  # noqa: E402
import nca  # noqa: E402
from train_bounce import MODES, Setup  # noqa: E402

CFG = bounce.BounceConfig()
ARGS = SimpleNamespace(channels=16, hidden=32, perception="sobel", no_compile=True)
LAT = torch.from_numpy(bounce.lattice_target(CFG))


@pytest.mark.parametrize("mode", MODES)
def test_histories_match_lattice(mode):
    st = Setup(mode, CFG, ARGS)
    hist = torch.arange(st.V) if mode == "independent" else None
    tgt = st.target_for(hist)
    hists = st.histories(tgt.expand(st.V if hist is not None else 1, *tgt.shape[1:]), hist)
    for v, h in enumerate(hists):
        assert torch.equal(h[0], LAT[:, v])


@pytest.mark.parametrize("mode", MODES)
def test_seed_sits_inside_the_ball_at_w0(mode):
    st = Setup(mode, CFG, ARGS)
    hist = torch.arange(st.V) if mode == "independent" else None
    seed = st.seed()
    x = seed.expand(st.V, *seed.shape) if hist is not None else seed[None]
    for h, t in zip(st.histories(x, hist), st.histories(st.target_for(hist).expand(x.shape[0], 4, *x.shape[2:]), hist)):
        alive = h[0, nca.ALPHA] > 0
        assert alive.sum() == 1 and alive[0].any()           # one live cell, at w = 0
        assert (t[0, nca.ALPHA][alive] > 0.9).all()          # inside the target ball


@pytest.mark.parametrize("mode", MODES)
def test_condition_is_zero_before_impact_and_e_after(mode):
    st = Setup(mode, CFG, ARGS)
    hist = torch.arange(st.V) if mode == "independent" else None
    c = st.cond(hist)
    full = c.expand(st.V if hist is not None else 1, 2, *c.shape[2:])
    # Put the condition in history form: (B, 2, W) per history.
    hs = st.histories(full, hist)
    for v, h in enumerate(hs):
        h = h[0, :, :, 0, 0, 0]                               # (2, W)
        assert torch.equal(h[:, : st.fork], torch.zeros(2, st.fork))
        assert torch.allclose(h[1, st.fork:], torch.full((st.W - st.fork,), CFG.bounciness[v]))
        assert torch.equal(h[0, st.fork:], torch.ones(st.W - st.fork))
