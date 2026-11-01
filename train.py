"""Train an NCA on an n-dimensional target (default: a 4D ball coloured red-to-blue along w).

Same recipe as train_lizard.py (pool training, per-parameter gradient normalisation,
lr drop), generalised to any n. Axis order of the grid is (w, z, y, x) for 4D; in
general the first spatial axis is the "extra" one that carries the colour gradient.

Outputs in --out: loss.jsonl, model.pt, atlas_*.png snapshots, and after training
growth_rgba.npy (uint8, shape (T, *grid, 4)) grown from a single seed, for the viewer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import nca

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------- targets

def _radius_grid(ndim: int, side: int) -> tuple[np.ndarray, np.ndarray]:
    c = (side - 1) / 2
    coords = np.stack(np.meshgrid(*[np.arange(side) - c] * ndim, indexing="ij"))
    return coords, np.sqrt((coords**2).sum(0))


def _colour(coords: np.ndarray, radius: float, alpha: np.ndarray) -> np.ndarray:
    """Red at low w, blue at high w; premultiplied by alpha."""
    t = np.clip(coords[0] / radius, -1, 1)  # -1 .. 1 along the first axis
    rgb = np.stack([(1 - t) / 2, 0.25 + 0 * t, (1 + t) / 2])
    return np.concatenate([rgb * alpha, alpha[None]]).astype(np.float32)


def make_target(kind: str, ndim: int, side: int, radius: float, thickness: float) -> torch.Tensor:
    coords, r = _radius_grid(ndim, side)
    if kind == "ball":
        alpha = np.clip(radius + 0.5 - r, 0, 1)  # anti-aliased edge
    elif kind == "shell":
        alpha = np.clip(thickness / 2 + 0.5 - np.abs(r - radius), 0, 1)
    else:
        raise ValueError(kind)
    return torch.from_numpy(_colour(coords, radius, alpha))  # (4, *grid)


def damage_masks(n: int, grid: tuple[int, ...], radius: float, rng: np.random.Generator) -> torch.Tensor:
    """n masks of shape (1, *grid), each zero inside a random ball that overlaps the target.
    Ball radius is 30-60% of the target radius; its centre is within one target radius
    of the grid centre on every axis (the n-D version of the original's damage circles)."""
    centre = (np.array(grid) - 1) / 2
    coords = np.stack(np.meshgrid(*[np.arange(s) for s in grid], indexing="ij"))
    masks = []
    for _ in range(n):
        c = centre + rng.uniform(-radius, radius, size=len(grid))
        r = rng.uniform(0.3, 0.6) * radius
        d = np.sqrt(((coords - c.reshape(-1, *[1] * len(grid))) ** 2).sum(0))
        masks.append(d > r)
    return torch.from_numpy(np.stack(masks)[:, None].astype(np.float32))


# ---------------------------------------------------------------- pictures

def atlas(x: torch.Tensor) -> np.ndarray:
    """Lay out every 2D (y, x) slice of one state as an RGB image.

    2D: the image. 3D: one row of z-slices. 4D: rows = w, columns = z.
    5D: one 4D atlas per v, side by side with a gap.
    """
    rgb = nca.to_rgb(x[None])[0].permute(*range(1, x.dim()), 0).detach().cpu().numpy()  # (*grid, 3)
    nd = rgb.ndim - 1
    if nd == 2:
        return rgb
    if nd == 3:
        return np.concatenate(list(rgb), axis=1)
    if nd == 4:
        return np.concatenate([np.concatenate(list(row), axis=1) for row in rgb], axis=0)
    if nd == 5:
        gap = np.ones((rgb.shape[1] * rgb.shape[3], rgb.shape[4], 3))
        tiles = []
        for v in range(rgb.shape[0]):
            tiles += [np.concatenate([np.concatenate(list(row), axis=1) for row in rgb[v]], axis=0), gap]
        return np.concatenate(tiles[:-1], axis=1)
    raise ValueError(nd)


def save_atlas(x: torch.Tensor, path: Path, scale: int = 4) -> None:
    img = Image.fromarray((atlas(x) * 255).astype(np.uint8))
    img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(path)


# ---------------------------------------------------------------- training

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ndim", type=int, default=4)
    p.add_argument("--side", type=int, default=12)
    p.add_argument("--target", choices=["ball", "shell"], default="ball")
    p.add_argument("--radius", type=float, default=None, help="default: side/2 - 1.5")
    p.add_argument("--thickness", type=float, default=2.0, help="shell thickness in cells")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--min-iters", type=int, default=24)
    p.add_argument("--max-iters", type=int, default=40)
    p.add_argument("--lr-drop", type=int, default=None, help="step at which lr drops 10x (default: steps*2/3)")
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--perception", choices=["sobel", "isotropic"], default="sobel")
    p.add_argument("--padding", choices=["zeros", "circular"], default="zeros")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--damage", type=int, default=0,
                   help="erase a random ball from this many of the best samples each step (regeneration training)")
    p.add_argument("--pool", type=int, default=256)
    p.add_argument("--dense", action="store_true", help="original rule: run the MLP on every cell (slower)")
    p.add_argument("--no-compile", action="store_true", help="skip torch.compile of perception")
    p.add_argument("--checkpoint", action="store_true", help="gradient checkpointing per CA step")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--snap-every", type=int, default=250)
    p.add_argument("--grow-steps", type=int, default=150, help="steps exported from a single seed at the end")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    radius = args.radius if args.radius is not None else args.side / 2 - 1.5
    out = args.out or ROOT / "runs" / f"{args.target}{args.ndim}d_s{args.side}"
    out.mkdir(parents=True, exist_ok=True)
    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    target = make_target(args.target, args.ndim, args.side, radius, args.thickness)
    grid = (args.side,) * args.ndim
    model = nca.NCA(args.ndim, args.channels, args.hidden, perception=args.perception, padding=args.padding,
                    skip_dead=not args.dense, compile=not args.no_compile)
    seed_state = model.seed(grid)[0]
    pool = seed_state.expand(args.pool, *seed_state.shape).clone()
    reduce_dims = tuple(range(1, args.ndim + 2))

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lr_drop = args.lr_drop or (2 * args.steps) // 3
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_drop], gamma=0.1)
    save_atlas(torch.cat([target, torch.zeros(args.channels - 4, *grid)]), out / "target.png")
    config = vars(args) | {"radius": radius, "out": str(out), "lr_drop": lr_drop}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(config, flush=True)

    log = open(out / "loss.jsonl", "a")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.choice(args.pool, args.batch, replace=False)
        x0 = pool[idx]
        with torch.no_grad():
            per_sample = ((nca.to_rgba(x0) - target) ** 2).mean(dim=reduce_dims)
        order = torch.argsort(per_sample, descending=True).numpy()
        idx, x0 = idx[order], x0[order]
        x0[0] = seed_state
        if args.damage:
            x0[-args.damage:] *= damage_masks(args.damage, grid, radius, rng)

        iters = int(rng.integers(args.min_iters, args.max_iters + 1))
        x = model.rollout(x0, iters, checkpoint=args.checkpoint)
        loss = ((nca.to_rgba(x) - target) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for prm in model.parameters():
                prm.grad /= prm.grad.norm() + 1e-8
        opt.step()
        sched.step()
        pool[idx] = x.detach()

        if step % args.log_every == 0 or step == 1:
            rec = {"step": step, "loss": loss.item(), "log10": round(float(np.log10(loss.item())), 3),
                   "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if step % args.snap_every == 0 or step == args.steps:
            save_atlas(x[0], out / f"atlas_{step:05d}_fresh.png")  # sample grown from a seed this round
            save_atlas(x[-1], out / f"atlas_{step:05d}_pool.png")
            torch.save({"model": model.state_dict(), "config": config, "step": step}, out / "model.pt")

    # Grow from one seed and export every step.
    with torch.no_grad():
        x = model.seed(grid)
        frames, losses = [x], []
        for _ in range(args.grow_steps):
            x = model(x)
            frames.append(x)
            losses.append(((nca.to_rgba(x) - target) ** 2).mean().item())
    frames = torch.cat(frames)
    rgba = nca.to_rgba(frames).clamp(0, 1).mul(255).byte()
    np.save(out / "growth_rgba.npy", rgba.permute(0, *range(2, args.ndim + 2), 1).numpy())
    for t in sorted(t for t in {10, 20, 40, args.grow_steps // 2, args.grow_steps} if t <= args.grow_steps):
        save_atlas(frames[t], out / f"grow_t{t:03d}.png")
    summary = {"grow_log10_loss": {t: round(float(np.log10(losses[t - 1])), 3)
                                   for t in (10, 20, 40, 80, args.grow_steps) if t <= args.grow_steps}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
