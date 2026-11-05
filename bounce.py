"""Exact bouncing-ball targets for comparing two ideas of a fifth dimension.

A ball (radius r) falls under gravity, drifts at constant speed along x, and bounces
on a floor, keeping a fraction e (its bounciness, the coefficient of restitution) of
its vertical speed at each impact. Its whole history is frozen into the w axis: each
w is one moment, so one history is a 4D shape (w, z, y, x). The first impact falls at
the same w for every e, so all histories are identical before it.

Three targets from one set of histories:
  lattice      5D grid (v, w, z, y, x), v indexing bounciness. Neighbouring v couple
               everywhere, so after the impact the histories fan out side by side.
  tree         w branches at the first impact: a shared trunk, then one branch per e.
               Stored as nodes, each a 3D grid, with a parent index.
  independent  the same histories with no coupling: the lattice's v slices, each
               trained on its own (no separate file needed).

Grid conventions match train.py: channel-first premultiplied RGBA, y increasing
downward (the floor is at the bottom of the y axis).

    .venv/bin/python bounce.py            # writes runs/targets/bounce_{lattice,tree}/
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
COLOUR = (1.0, 0.55, 0.15)  # one colour for every history, so pre-impact slices match exactly


@dataclass
class BounceConfig:
    X: int = 16                 # grid size along x (drift direction)
    Y: int = 12                 # along y (vertical; floor at the bottom)
    Z: int = 8                  # along z (depth; the ball stays centred)
    W: int = 24                 # moments in each history
    radius: float = 2.0
    drop_height: float = 9.0    # initial height of the ball's centre above the floor, in cells
    impact_w: float = 8.0       # w (in moments) at which the ball first hits the floor
    x_start: float = 2.5        # x of the centre at w = 0
    x_end: float = 13.5         # x of the centre at w = W - 1
    bounciness: list[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 0.9])

    @property
    def gravity(self) -> float:
        """Chosen so the first impact lands at w = impact_w."""
        return 2 * (self.drop_height - self.radius) / self.impact_w**2


def height(t: float, e: float, cfg: BounceConfig) -> float:
    """Height of the ball's centre above the floor at time t (one time unit = one w step)."""
    g, r = cfg.gravity, cfg.radius
    t_hit = math.sqrt(2 * (cfg.drop_height - r) / g)
    if t <= t_hit:
        return cfg.drop_height - 0.5 * g * t * t
    v = e * g * t_hit                     # upward speed just after the first bounce
    t0 = t_hit
    while v > 1e-6:
        flight = 2 * v / g
        if t <= t0 + flight:
            s = t - t0
            return r + v * s - 0.5 * g * s * s
        t0 += flight
        v *= e
    return r                              # came to rest


def first_impact_index(cfg: BounceConfig) -> int:
    """Number of moments at or before the first impact: the length of the shared trunk."""
    t_hit = math.sqrt(2 * (cfg.drop_height - cfg.radius) / cfg.gravity)
    return int(math.floor(t_hit)) + 1


def render_history(e: float, cfg: BounceConfig) -> np.ndarray:
    """One bounce history as a (4, W, Z, Y, X) premultiplied-RGBA float32 array."""
    zc = (cfg.Z - 1) / 2
    z, y, x = np.meshgrid(np.arange(cfg.Z), np.arange(cfg.Y), np.arange(cfg.X), indexing="ij")
    out = np.zeros((4, cfg.W, cfg.Z, cfg.Y, cfg.X), np.float32)
    for w in range(cfg.W):
        xc = cfg.x_start + (cfg.x_end - cfg.x_start) * w / (cfg.W - 1)
        yc = (cfg.Y - 0.5) - height(float(w), e, cfg)   # floor surface is the bottom edge of row Y-1
        d = np.sqrt((x - xc) ** 2 + (y - yc) ** 2 + (z - zc) ** 2)
        a = np.clip(cfg.radius + 0.5 - d, 0, 1)
        out[:3, w] = np.array(COLOUR, np.float32)[:, None, None, None] * a
        out[3, w] = a
    return out


def lattice_target(cfg: BounceConfig) -> np.ndarray:
    """(4, V, W, Z, Y, X): one history per bounciness value along v."""
    return np.stack([render_history(e, cfg) for e in cfg.bounciness], axis=1)


def tree_target(cfg: BounceConfig, lattice: np.ndarray | None = None) -> dict:
    """The same histories as a tree along w.

    Returns nodes (N, 4, Z, Y, X); parent (N,) with -1 for the root; node_w (N,);
    node_branch (N,) with -1 on the trunk. Trunk nodes come first (w = 0 .. fork-1),
    then each branch's nodes in w order. Built from the lattice, so the two match exactly.
    """
    lat = lattice_target(cfg) if lattice is None else lattice
    fork = first_impact_index(cfg)
    if not np.array_equal(lat[:, :, :fork].min(axis=1), lat[:, :, :fork].max(axis=1)):
        raise AssertionError("histories differ before the first impact")
    nodes, parent, node_w, node_branch = [], [], [], []
    for w in range(fork):
        nodes.append(lat[:, 0, w])
        parent.append(w - 1)
        node_w.append(w)
        node_branch.append(-1)
    for b in range(len(cfg.bounciness)):
        prev = fork - 1
        for w in range(fork, cfg.W):
            nodes.append(lat[:, b, w])
            parent.append(prev)
            prev = len(nodes) - 1
            node_w.append(w)
            node_branch.append(b)
    return {"nodes": np.stack(nodes), "parent": np.array(parent), "node_w": np.array(node_w),
            "node_branch": np.array(node_branch), "fork_w": fork}


def tree_edges(parent: np.ndarray) -> np.ndarray:
    """(E, 2) array of (parent, child) pairs: the links a tree NCA couples along w."""
    child = np.nonzero(parent >= 0)[0]
    return np.stack([parent[child], child], axis=1)


# ---------------------------------------------------------------- export

def to_uint8(rgba_first: np.ndarray) -> np.ndarray:
    """(4, *grid) float -> (*grid, 4) uint8, the viewer's layout."""
    return (np.moveaxis(np.clip(rgba_first, 0, 1), 0, -1) * 255).round().astype(np.uint8)


def fan_image(lat: np.ndarray, scale: int = 4) -> Image.Image:
    """Side view of every history: rows = bounciness, columns = w; each tile is the
    (y, x) view projected through z, on white."""
    V, W = lat.shape[1], lat.shape[2]
    a = lat[3].max(axis=2)                                   # (V, W, Y, X)
    rgb = lat[:3].max(axis=3)                                # (3, V, W, Y, X)
    img = np.clip(1 - a[None] + rgb, 0, 1)
    tiles = img.transpose(1, 2, 3, 4, 0)                     # (V, W, Y, X, 3)
    Y, X = tiles.shape[2:4]
    canvas = np.full((V * (Y + 1) - 1, W * (X + 1) - 1, 3), 0.85, np.float32)
    for v in range(V):
        for w in range(W):
            canvas[v * (Y + 1): v * (Y + 1) + Y, w * (X + 1): w * (X + 1) + X] = tiles[v, w]
    im = Image.fromarray((canvas * 255).astype(np.uint8))
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=ROOT / "runs" / "targets")
    args = p.parse_args()
    cfg = BounceConfig()
    lat = lattice_target(cfg)
    tree = tree_target(cfg, lat)

    d = args.out / "bounce_lattice"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "target_rgba.npy", to_uint8(lat)[None])              # (1, V, W, Z, Y, X, 4)
    (d / "config.json").write_text(json.dumps(asdict(cfg) | {"axes": ["v", "w", "z", "y", "x"],
                                                            "fork_w": tree["fork_w"]}, indent=2))
    fan_image(lat).save(d / "fan.png")

    d = args.out / "bounce_tree"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "target_rgba.npy", to_uint8(np.moveaxis(tree["nodes"], 0, 1))[None])  # (1, N, Z, Y, X, 4)
    (d / "tree.json").write_text(json.dumps({
        "parent": tree["parent"].tolist(), "node_w": tree["node_w"].tolist(),
        "node_branch": tree["node_branch"].tolist(), "fork_w": tree["fork_w"],
        "branch_labels": [f"bounciness {e}" for e in cfg.bounciness]}))
    (d / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    cells_lat = lat[0].size
    cells_tree = tree["nodes"][:, 0].size
    print(f"lattice: grid {lat.shape[1:]} = {cells_lat:,} cells")
    print(f"tree:    {len(tree['parent'])} nodes ({tree['fork_w']} trunk + {len(cfg.bounciness)} branches x "
          f"{cfg.W - tree['fork_w']}) of {cfg.Z}x{cfg.Y}x{cfg.X} = {cells_tree:,} cells")
    print(f"independent: {len(cfg.bounciness)} x 4D grids of {lat.shape[2:]} = {cells_lat:,} cells, no coupling")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
