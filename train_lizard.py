"""Reproduce the Growing NCA lizard (Distill 2020, experiment 2: "persistent", pool training)
with the n-generic NCA at n = 2. This is the check that the n-D code is a faithful port
before going up to 4D.

Hyperparameters follow the original colab: 16 channels, 128 hidden units, fire rate 0.5,
40 px target padded by 16, pool of 1024, batch 8, 64-96 steps per sample, Adam with
lr 2e-3 dropping to 2e-4 at step 2000, per-parameter gradient normalisation.
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


def load_target(path: Path, size: int = 40, pad: int = 16) -> torch.Tensor:
    img = Image.open(path).convert("RGBA")
    img.thumbnail((size, size), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a[..., :3] *= a[..., 3:]  # premultiply
    a = np.pad(a, ((pad, pad), (pad, pad), (0, 0)))
    return torch.from_numpy(a).permute(2, 0, 1)  # (4, H, W)


def save_png(x: torch.Tensor, path: Path, scale: int = 4) -> None:
    """Save a row of states (B, C, H, W) as one image, composited over white."""
    rgb = nca.to_rgb(x).permute(0, 2, 3, 1).detach().cpu().numpy()
    row = np.concatenate(list(rgb), axis=1)
    img = Image.fromarray((row * 255).astype(np.uint8))
    img.resize((img.width * scale, img.height * scale), Image.NEAREST).save(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "lizard2d")
    p.add_argument("--target", type=Path, default=ROOT / "data" / "lizard_128.png")
    p.add_argument("--perception", choices=["sobel", "isotropic"], default="sobel")
    p.add_argument("--padding", choices=["zeros", "circular"], default="zeros")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--pool", type=int, default=1024)
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--snap-every", type=int, default=500)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    target = load_target(args.target)
    shape = tuple(target.shape[1:])
    model = nca.NCA(ndim=2, perception=args.perception, padding=args.padding)
    seed_state = model.seed(shape)[0]
    pool = seed_state.expand(args.pool, *seed_state.shape).clone()

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2000], gamma=0.1)
    save_png(target[None].expand(1, -1, -1, -1).repeat(1, 4, 1, 1)[:, :16], args.out / "target.png")

    log = open(args.out / "loss.jsonl", "a")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = np.random.choice(args.pool, args.batch, replace=False)
        x0 = pool[idx]
        # Replace the worst sample with a fresh seed so the model keeps learning to grow.
        with torch.no_grad():
            per_sample = ((nca.to_rgba(x0) - target) ** 2).mean(dim=(1, 2, 3))
        order = torch.argsort(per_sample, descending=True)
        idx, x0 = idx[order.numpy()], x0[order]
        x0[0] = seed_state

        iters = int(np.random.randint(64, 97))
        x = model.rollout(x0, iters)
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
            rec = {"step": step, "loss": loss.item(), "log10": float(np.log10(loss.item())), "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if step % args.snap_every == 0 or step == args.steps:
            save_png(x, args.out / f"batch_{step:05d}.png")
            torch.save({"model": model.state_dict(), "args": vars(args) | {"out": str(args.out), "target": str(args.target)}, "step": step},
                       args.out / "model.pt")

    # Grow from a single seed and export every step for the viewer.
    with torch.no_grad():
        x = model.seed(shape)
        frames = [x]
        for _ in range(200):
            x = model(x)
            frames.append(x)
    frames = torch.cat(frames)
    np.save(args.out / "growth_rgba.npy", nca.to_rgba(frames).clamp(0, 1).mul(255).byte().permute(0, 2, 3, 1).numpy())
    save_png(frames[::25], args.out / "growth.png")
    final = ((nca.to_rgba(frames[-1:]) - target) ** 2).mean().item()
    print({"seed_to_200_steps_loss": final, "log10": float(np.log10(final))})


if __name__ == "__main__":
    main()
