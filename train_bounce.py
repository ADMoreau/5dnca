"""Train an NCA to grow the bouncing-ball histories (bounce.py), one of four ways:

  lattice      5D NCA on (v, w, z, y, x); neighbouring bounciness values couple everywhere.
  tree         4D NCA whose w axis forks at the first impact (tree_nca.TreeNCA), two-way links.
  tree-down    the same tree with one-way links (nothing flows from branches back to the trunk).
  independent  4D NCA on one history at a time; histories never touch.

Every mode grows from one live cell at the ball's starting position (w = 0) and uses
the same recipe as train.py (pool training, per-parameter gradient normalisation,
lr drop). The tree's branches are mirror images of each other, so nothing in its
structure says which branch is which. Every mode therefore gets the same two fixed
input channels per cell: "after the first impact" (0/1) and, when it is 1, the
bounciness e of that cell's history. Before the impact every cell sees (0, 0).

Outputs in --out: config.json, loss.jsonl, model.pt, snapshot images, eval.json
(per-history error at several growth lengths, the shared yardstick for comparing
modes) and growth_rgba.npy for the viewer (plus tree.json in tree modes).

    .venv/bin/python train_bounce.py --mode tree --steps 1500
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import bounce
import nca
from tree_nca import TreeNCA

ROOT = Path(__file__).resolve().parent
MODES = ("lattice", "tree", "tree-down", "independent")


class Setup:
    """Model, target, condition inputs and seed for one mode. All tensors are channel-first."""

    def __init__(self, mode: str, cfg: bounce.BounceConfig, args):
        self.mode, self.cfg = mode, cfg
        lat = torch.from_numpy(bounce.lattice_target(cfg))           # (4, V, W, Z, Y, X)
        self.V, self.W = lat.shape[1], lat.shape[2]
        self.fork = bounce.first_impact_index(cfg)
        e = torch.tensor(cfg.bounciness, dtype=torch.float32)
        after = (torch.arange(self.W) >= self.fork).float()           # (W,)
        kw = dict(channels=args.channels, hidden=args.hidden, perception=args.perception,
                  padding="zeros", skip_dead=True, compile=not args.no_compile, cond_channels=2)
        # Seed: the cell nearest the ball's centre at w = 0.
        zc, yc = (cfg.Z - 1) / 2, (cfg.Y - 0.5) - cfg.drop_height
        self.seed_zyx = (int(zc), int(round(yc)), int(round(cfg.x_start)))

        if mode == "lattice":
            self.model = nca.NCA(5, **kw)
            self.target = lat
            self.grid = tuple(lat.shape[1:])
            cond = torch.stack([after.expand(self.V, -1), e[:, None] * after])  # (2, V, W)
            self.cond_all = cond[None, :, :, :, None, None, None]
        elif mode in ("tree", "tree-down"):
            tree = bounce.tree_target(cfg, lat.numpy())
            self.tree = tree
            self.model = TreeNCA(tree["parent"], edges="both" if mode == "tree" else "down", **kw)
            self.target = torch.from_numpy(tree["nodes"]).movedim(0, 1)  # (4, N, Z, Y, X)
            self.grid = tuple(self.target.shape[1:])
            br = torch.from_numpy(tree["node_branch"])
            known = (br >= 0).float()
            cond = torch.stack([known, e[br.clamp(min=0)] * known])     # (2, N)
            self.cond_all = cond[None, :, :, None, None, None]
        elif mode == "independent":
            self.model = nca.NCA(4, **kw)
            self.target = lat.movedim(1, 0)                             # (V, 4, W, Z, Y, X)
            self.grid = tuple(lat.shape[2:])
            self.cond_v = torch.stack([after.expand(self.V, -1), e[:, None] * after], dim=1)  # (V, 2, W)
            self.cond_v = self.cond_v[:, :, :, None, None, None]
        else:
            raise ValueError(mode)

    def seed(self) -> torch.Tensor:
        x = torch.zeros(self.model.channels, *self.grid)
        if self.mode == "lattice":
            x[(slice(nca.ALPHA, None), slice(None), 0) + self.seed_zyx] = 1.0   # every v at w = 0
        else:
            x[(slice(nca.ALPHA, None), 0) + self.seed_zyx] = 1.0                # w = 0 / the root
        return x

    def cond(self, hist: torch.Tensor | None) -> torch.Tensor:
        """Condition inputs for a batch; `hist` is each sample's history index (independent mode)."""
        return self.cond_v[hist] if self.mode == "independent" else self.cond_all

    def target_for(self, hist: torch.Tensor | None) -> torch.Tensor:
        return self.target[hist] if self.mode == "independent" else self.target[None]

    def histories(self, x: torch.Tensor, hist: torch.Tensor | None = None) -> list[torch.Tensor]:
        """Split states into the V histories, each (B, C, W, Z, Y, X)."""
        if self.mode == "lattice":
            return [x[:, :, v] for v in range(self.V)]
        if self.mode == "independent":
            return [x[hist == v] for v in range(self.V)]
        return [x[:, :, self.path(b)] for b in range(self.V)]

    def path(self, b: int) -> list[int]:
        nb = self.W - self.fork
        return list(range(self.fork)) + list(range(self.fork + b * nb, self.fork + (b + 1) * nb))

    def cells(self) -> int:
        n = int(np.prod(self.grid))
        return n * self.V if self.mode == "independent" else n


def side_view(hists: list[torch.Tensor], scale: int = 4) -> Image.Image:
    """Rows = histories, columns = w; each tile is the (y, x) view projected through z, on white."""
    rows = []
    for h in hists:                                                   # (C, W, Z, Y, X)
        a = h[nca.ALPHA].clamp(0, 1).amax(dim=1)                      # (W, Y, X)
        rgb = h[:3].clamp(0, 1).amax(dim=2)                           # (3, W, Y, X)
        img = (1 - a[None] + rgb).clamp(0, 1).permute(1, 2, 3, 0)     # (W, Y, X, 3)
        W, Y, X = img.shape[:3]
        row = torch.full((Y, W * (X + 1) - 1, 3), 0.85)
        for w in range(W):
            row[:, w * (X + 1): w * (X + 1) + X] = img[w]
        rows += [row, torch.full((1, row.shape[1], 3), 0.85)]
    canvas = torch.cat(rows[:-1]).numpy()
    im = Image.fromarray((canvas * 255).astype(np.uint8))
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def grow(st: Setup, steps: list[int], record_every: int = 0):
    """Grow every history from its seed; per-history log10 MSE at each step in `steps`.
    Returns (errors {step: [V values]}, frames as a list of lattice-shaped (V, C, W, Z, Y, X) or
    native states)."""
    model = st.model
    with torch.no_grad():
        if st.mode == "independent":
            hist = torch.arange(st.V)
            x = st.seed().expand(st.V, *st.seed().shape).clone()
        else:
            hist = None
            x = st.seed()[None]
        cond = st.cond(hist)
        target = st.target_for(hist)
        tgt_h = st.histories(target.expand(x.shape[0], *target.shape[1:]), hist)
        errs, frames = {}, [x[0] if hist is None else x] if record_every else []
        for t in range(1, max(steps) + 1):
            x = model(x, cond=cond)
            if t in steps:
                errs[t] = [round(float(torch.log10(((nca.to_rgba(h) - th) ** 2).mean())), 3)
                           for h, th in zip(st.histories(x, hist), tgt_h)]
            if record_every and t % record_every == 0:
                frames.append(x[0] if hist is None else x)
    return errs, frames, x, hist


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--min-iters", type=int, default=64)
    p.add_argument("--max-iters", type=int, default=96)
    p.add_argument("--lr-drop", type=int, default=None, help="default: steps*2/3")
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--perception", choices=["sobel", "isotropic"], default="sobel")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--pool", type=int, default=256)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-checkpoint", action="store_true", help="keep every step's activations (more memory)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--snap-every", type=int, default=250)
    p.add_argument("--grow-steps", type=int, default=300, help="steps recorded from a single seed at the end")
    p.add_argument("--bench", type=int, default=0, help="time this many training steps and exit")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = bounce.BounceConfig()
    st = Setup(args.mode, cfg, args)
    model = st.model
    out = args.out or ROOT / "runs" / f"bounce_{args.mode}"
    if not args.bench:
        out.mkdir(parents=True, exist_ok=True)

    seed_state = st.seed()
    pool = seed_state.expand(args.pool, *seed_state.shape).clone()
    pool_hist = torch.arange(args.pool) % st.V if args.mode == "independent" else None
    reduce_dims = tuple(range(1, seed_state.dim() + 1))

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lr_drop = args.lr_drop or (2 * args.steps) // 3
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_drop], gamma=0.1)
    config = vars(args) | {"out": str(out), "lr_drop": lr_drop, "bounce": cfg.__dict__, "grid": st.grid,
                           "fork_w": st.fork, "seed_zyx": st.seed_zyx, "cells": st.cells()}
    config = json.loads(json.dumps(config, default=str))
    print(config, flush=True)
    if not args.bench:
        (out / "config.json").write_text(json.dumps(config, indent=2))
        side_view(sample_hists(st, st.target_for(torch.arange(st.V)) if args.mode == "independent"
                               else st.target[None])).save(out / "target.png")
    log = None if args.bench else open(out / "loss.jsonl", "a")

    t0 = time.time()
    for step in range(1, (args.bench or args.steps) + 1):
        idx = torch.from_numpy(rng.choice(args.pool, args.batch, replace=False))
        hist = pool_hist[idx] if pool_hist is not None else None
        x0 = pool[idx]
        target = st.target_for(hist)
        with torch.no_grad():
            per_sample = ((nca.to_rgba(x0) - target) ** 2).mean(dim=reduce_dims)
        order = torch.argsort(per_sample, descending=True)
        idx, x0 = idx[order], x0[order]
        hist = pool_hist[idx] if pool_hist is not None else None
        target = st.target_for(hist)
        x0[0] = seed_state

        iters = int(rng.integers(args.min_iters, args.max_iters + 1))
        x = model.rollout(x0, iters, checkpoint=not args.no_checkpoint, cond=st.cond(hist))
        loss = ((nca.to_rgba(x) - target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for prm in model.parameters():
                prm.grad /= prm.grad.norm() + 1e-8
        opt.step()
        sched.step()
        pool[idx] = x.detach()

        if args.bench:
            print({"step": step, "iters": iters, "sec": round(time.time() - t0, 1),
                   "log10": round(float(np.log10(loss.item())), 3)}, flush=True)
            continue
        if step % args.log_every == 0 or step == 1:
            rec = {"step": step, "loss": loss.item(), "log10": round(float(np.log10(loss.item())), 3),
                   "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if step % args.snap_every == 0 or step == args.steps:
            side_view(sample_hists(st, x.detach())).save(out / f"snap_{step:05d}.png")
            torch.save({"model": model.state_dict(), "config": config, "step": step}, out / "model.pt")
    if args.bench:
        return

    # Grow from one seed: per-history error over time, and a recording for the viewer.
    checks = sorted({50, 100, 150, 200, args.grow_steps, 2 * args.grow_steps})
    errs, frames, final, hist = grow(st, checks, record_every=2)
    (out / "eval.json").write_text(json.dumps({"log10_mse_per_history": errs,
                                               "bounciness": cfg.bounciness}, indent=2))
    print({"log10_mse_per_history": errs}, flush=True)
    frames = frames[: args.grow_steps // 2 + 1]
    stack = torch.stack([f if args.mode != "independent" else f.movedim(0, 1) for f in frames])  # (T, C, *grid)
    rgba = nca.to_rgba(stack).clamp(0, 1).mul(255).byte()
    np.save(out / "growth_rgba.npy", rgba.movedim(1, -1).numpy())
    if args.mode in ("tree", "tree-down"):
        shutil.copy(ROOT / "runs" / "targets" / "bounce_tree" / "tree.json", out / "tree.json")
    side_view(sample_hists(st, final)).save(out / f"grown_t{max(checks)}.png")


def sample_hists(st: Setup, x: torch.Tensor) -> list[torch.Tensor]:
    """Histories to draw, each (C, W, Z, Y, X): every history of the first sample, or in
    independent mode every sample (each one is a single history)."""
    if st.mode == "independent":
        return list(x)
    return [h[0] for h in st.histories(x[:1])]


if __name__ == "__main__":
    main()
