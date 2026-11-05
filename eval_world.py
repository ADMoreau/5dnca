"""Run a trained NCA in a world bigger than its training grid, with several seeds.

Checks the two things the training grid can't show:
  1. Each seed grows one shape of the trained size and then stops; growth doesn't
     fill the world (a model that learned to stop at the training grid's walls fails here).
  2. After part of one shape is erased, it regrows.

Saves a recording (every --every steps) as <run>/world/growth_rgba.npy so the viewer
lists it, plus world/summary.json.

    .venv/bin/python eval_world.py runs/ball4d_r35_wrap
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import nca
import train


def load_model(run: Path) -> tuple[nca.NCA, dict]:
    ck = torch.load(run / "model.pt")
    cfg = ck.get("config") or ck["args"]
    model = train.build_model(cfg, "zeros", compile=cfg.get("ndim", 2) >= 3)
    model.load_state_dict(ck["model"])
    return model.eval(), cfg


def components(alive: torch.Tensor) -> int:
    """Number of connected groups of live cells (3^n neighbourhood), by label propagation."""
    labels = torch.arange(1, alive.numel() + 1, dtype=torch.float32).reshape(alive.shape) * alive
    while True:
        new = nca.max_pool3(labels[None, None])[0, 0] * alive
        if torch.equal(new, labels):
            break
        labels = new
    return int(torch.unique(labels[alive.bool()]).numel())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--world", default=None, help="comma-separated world shape (default: extra axes as trained, z/y/x = 24)")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--damage-at", type=int, default=200, help="erase half of the first shape at this step (0 = never)")
    p.add_argument("--every", type=int, default=3, help="record every Nth step")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, cfg = load_model(args.run)
    nd = model.ndim
    side = cfg.get("side", 72)
    if args.world:
        world = tuple(int(s) for s in args.world.split(","))
    else:
        world = tuple([side] * max(nd - 3, 0) + [24] * min(nd, 3))
    assert len(world) == nd, f"world must have {nd} axes"

    radius = cfg.get("radius", side / 2 - 1.5)
    target = train.make_target(cfg.get("target", "ball"), nd, side, radius, cfg.get("thickness", 2.0))
    one_shape_alpha = float(target[3].sum())

    # Three seeds in a triangle on the middle z plane, centred on the extra axes.
    mid = [s // 2 for s in world]
    sp = max(int(round(2 * radius + 3)), 4)  # centre spacing: one diameter plus a small gap
    offsets = [(-sp // 2, -sp // 2), (-sp // 2, sp - sp // 2), (sp - sp // 2, 0)]
    seeds = []
    for dy, dx in offsets:
        pos = list(mid)
        pos[-2] += dy
        pos[-1] += dx
        seeds.append(tuple(pos))

    out = args.run / "world"
    out.mkdir(exist_ok=True)
    x = torch.zeros(1, model.channels, *world)
    for pos in seeds:
        x[(0, slice(nca.ALPHA, None)) + pos] = 1.0

    frames, log = [], []
    t0 = time.time()
    with torch.no_grad():
        for step in range(args.steps + 1):
            if step and step == args.damage_at:
                # Erase the half of the first shape on the -x side of its seed.
                coords = torch.stack(torch.meshgrid(*[torch.arange(s) for s in world], indexing="ij"))
                c = torch.tensor(seeds[0]).reshape(-1, *[1] * nd)
                in_shape = ((coords - c) ** 2).sum(0).sqrt() <= radius + 1.5
                x = x * ~(in_shape & (coords[-1] < seeds[0][-1]))
            if step:
                x = model(x)
            if step % args.every == 0:
                frames.append(nca.to_rgba(x)[0].clamp(0, 1).mul(255).byte().permute(*range(1, nd + 1), 0).numpy())
            if step % 25 == 0 or step in (args.damage_at - 1, args.damage_at):
                a = x[0, nca.ALPHA].clamp(0, 1)
                alive = (a > 0.1).float()
                rec = {"step": step, "alpha_vs_3_shapes": round(float(a.sum()) / (len(seeds) * one_shape_alpha), 3),
                       "components": components(alive) if alive.sum() else 0,
                       "touches_world_edge": bool(sum(alive.narrow(d, 0, 1).sum() + alive.narrow(d, -1, 1).sum()
                                                       for d in range(max(nd - 3, 0), nd)) > 0)}
                log.append(rec)
                print(rec, flush=True)

    np.save(out / "growth_rgba.npy", np.stack(frames))
    summary = {"world": world, "seeds": seeds, "one_shape_alpha": round(one_shape_alpha, 1),
               "damage_at": args.damage_at, "record_every": args.every,
               "sec_per_step": round((time.time() - t0) / args.steps, 3), "log": log}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print({k: v for k, v in summary.items() if k != "log"})


if __name__ == "__main__":
    main()
