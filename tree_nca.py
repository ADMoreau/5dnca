"""A 4D NCA whose w axis is a tree instead of a line.

The state has the same layout as a 4D grid, (batch, channels, node, z, y, x), but
neighbours along the node axis come from a parent array instead of index +-1:

    left neighbour  = the parent       (zero at the root, like zero padding)
    right neighbour = mean of children (zero at a leaf)

Everything else is the 4D rule: the Sobel along w is a central difference between
those two neighbours with [1, 2, 1] smoothing on z, y and x, and the z/y/x Sobels are
smoothed along w with (parent + 2 * self + children) / 4. On a chain (each node has one
child) this is exactly nca.NCA(ndim=4), so weights move freely between the two.

edges="both" (the default) lets information flow both ways along w: a branch's
contents can reshape the trunk. edges="down" makes the links one-way: a cell reads
only its parent and itself along w, so nothing flows back from children to parents.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn.functional as F

import nca

warnings.filterwarnings("ignore", message="index_reduce", category=UserWarning)  # beta API, used under no_grad


def pad_spatial(x: torch.Tensor, padding: str = "zeros", fill: float = 0.0) -> torch.Tensor:
    """Pad z, y and x (the last three axes) by one cell; leave the node axis alone."""
    if padding == "zeros":
        return F.pad(x, (1, 1, 1, 1, 1, 1), value=fill)
    if padding == "circular":
        for d in (-3, -2, -1):
            x = torch.cat([x.narrow(d, -1, 1), x, x.narrow(d, 0, 1)], dim=d)
        return x
    raise ValueError(f"unknown padding {padding!r}")


class TreeNCA(nca.NCA):
    def __init__(self, parent, edges: str = "both", **kw):
        if edges not in ("both", "down"):
            raise ValueError(f"unknown edges {edges!r}")
        compile_ = kw.pop("compile", False)
        super().__init__(ndim=4, compile=False, **kw)
        parent = torch.as_tensor(np.asarray(parent), dtype=torch.long)
        n = len(parent)
        child = torch.nonzero(parent >= 0).squeeze(1)
        count = torch.bincount(parent[child], minlength=n)
        self.edges = edges
        self.n_nodes = n
        # Not saved in the state dict, so the weights load into nca.NCA(ndim=4) and back.
        self.register_buffer("parent", parent.clamp(min=0), persistent=False)
        self.register_buffer("has_parent", (parent >= 0).float().view(1, 1, n, 1, 1, 1), persistent=False)
        self.register_buffer("child", child, persistent=False)
        self.register_buffer("child_parent", parent[child], persistent=False)
        self.register_buffer("inv_count", (1.0 / count.clamp(min=1)).float().view(1, 1, n, 1, 1, 1),
                             persistent=False)
        self._percept = torch.compile(self._perceive_tree, dynamic=False) if compile_ else self._perceive_tree

    # ---- neighbours along w
    def up(self, x: torch.Tensor) -> torch.Tensor:
        """Each node's parent value; zero at the root."""
        return x.index_select(2, self.parent) * self.has_parent

    def down(self, x: torch.Tensor) -> torch.Tensor:
        """Mean of each node's children; zero at a leaf. With one-way edges, the node itself."""
        if self.edges == "down":
            return x
        s = torch.zeros_like(x).index_add(2, self.child_parent, x.index_select(2, self.child))
        return s * self.inv_count

    # ---- perception and alive mask
    def _perceive_tree(self, x: torch.Tensor) -> torch.Tensor:
        xp = pad_spatial(x, self.padding)
        lft, rgt = self.up(xp), self.down(xp)
        gw = (rgt - lft) * 0.5
        for a in (1, 2, 3):
            gw = nca._smooth(gw, a)
        g = nca._sobels((lft + rgt + 2 * xp) * 0.25, [1, 2, 3])
        grads = [gw, g[1], g[2], g[3]]
        if self.perception == "sobel":
            return torch.cat([x] + grads, dim=1)
        if self.perception == "isotropic":
            core = xp[:, :, :, 1:-1, 1:-1, 1:-1]
            lap = lft[:, :, :, 1:-1, 1:-1, 1:-1] + rgt[:, :, :, 1:-1, 1:-1, 1:-1] - 8 * core
            for d in (3, 4, 5):
                t = xp
                for e in (3, 4, 5):
                    if e != d:
                        t = t.narrow(e, 1, xp.shape[e] - 2)
                lap = lap + t.narrow(d, 0, t.shape[d] - 2) + t.narrow(d, 2, t.shape[d] - 2)
            grad = torch.sqrt(sum(t * t for t in grads) + 1e-8)
            return torch.cat([x, lap, grad], dim=1)
        raise ValueError(f"unknown perception mode {self.perception!r}")

    def perceive_cells(self, x: torch.Tensor) -> torch.Tensor:
        return self._percept(x)

    def alive(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            a = pad_spatial(x[:, nca.ALPHA : nca.ALPHA + 1], self.padding, fill=float("-inf"))
            for d in (3, 4, 5):
                n = a.shape[d] - 2
                a = torch.maximum(torch.maximum(a.narrow(d, 0, n), a.narrow(d, 1, n)), a.narrow(d, 2, n))
            par = torch.where(self.has_parent.bool(), a.index_select(2, self.parent), float("-inf"))
            m = torch.maximum(a, par)
            if self.edges == "both":
                kids = torch.full_like(a, float("-inf")).index_reduce_(
                    2, self.child_parent, a.index_select(2, self.child), "amax")
                m = torch.maximum(m, kids)
            return m > 0.1

    def forward(self, x, fire_rate=None, cond=None):
        assert x.shape[2] == self.n_nodes, f"expected {self.n_nodes} nodes, got {x.shape[2]}"
        return super().forward(x, fire_rate, cond)
