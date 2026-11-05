"""A hierarchical NCA: the n-D rule plus coarser copies of the grid, so a cell hears from
far beyond its neighbours in a single step.

Level 0 is the ordinary grid. Level l halves the pooled axes l times (rounding up). Every
step is one sweep up and back down:

  up    each coarse cell gets the mean and the maximum of the fine cells beneath it.
        Nothing is learned on the way up; the fine cells learn what to put in their channels.
  down  from the coarsest level to the finest, each coarse level runs a small per-cell
        network on its neighbourhood of summaries plus what the level above passed down,
        and passes `context` channels on, interpolated linearly to the next finer level.
  fine  the ordinary NCA step, with the context from level 1 as extra inputs.

The coarse levels keep no state between steps: they are recomputed from the fine cells
every step. The state is therefore still one grid with the usual layout, so the training
scripts, sample pools, damage masks and the viewer work unchanged, and only fine cells
remember anything. Growth is still local, because the alive mask is untouched: a dead
cell comes alive only next to a living one, and the context only changes cells that are
already alive.

`levels` caps how far influence travels per step. In each direction it is at least
2**levels - 1 cells and at most 3 * 2**(levels - 1) - 2, depending on where a cell sits in
its blocks: 1, 3-4, 7-10, 15-22 cells for levels 1-4 (tests/test_hier_nca.py checks this).
The flat NCA is levels=1. Enough levels to shrink the grid to a single cell makes every
step global.

One-way time is kept by never pooling the causal axis. Every coarse level keeps all the
frames, and its perception is one-way along time like the fine level's.

The coarse networks start at zero, so a new HierNCA computes exactly what the flat NCA
does until training gives the context something to say.

Fixed blocks are a coordinate grid. Trained with the blocks in the same place every step,
the first 4D ball model grew its ball at the nearest multiple of 4 cells (the coarsest
block) instead of at its seed. With shift_blocks (the default) every step moves the blocks
by a random offset along each pooled axis, so where the blocks fall says nothing about
where a cell is. The offsets come from torch's generator, so a checkpointed rollout redraws
the same ones in its backward pass.

The training box must also be larger than what the coarsest level sees (3 blocks: 12 cells
with 3 levels). In a 12^4 box every coarse cell held part of the ball, so the coarse levels
never saw empty space, and in a bigger world the ball grew to only 82-86% of its size. In a
16^4 box with moving blocks it grows to full size anywhere.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import nca


_MAX_POOL = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}


def pool2(x: torch.Tensor, axes, mode: str) -> torch.Tensor:
    """Halve each spatial axis in `axes` (rounding up) by the mean or the maximum of each pair
    of cells. An odd length repeats its last cell first, so the last pair's statistic covers
    only the real cell."""
    if mode == "max":
        # PyTorch's max-pool kernels remember where each maximum came from, which makes the
        # backward pass about 5x faster than reducing pairs with amax. They cover up to the
        # last three axes; any other pooled axis is reduced pairwise on the smaller result.
        nd = x.dim() - 2
        k = 0
        while k < min(3, nd) and nd - 1 - k in axes:
            k += 1
        if k:
            tail = x.shape[-k:]
            size = tuple(1 if n == 1 else 2 for n in tail)
            y = _MAX_POOL[k](x.reshape(x.shape[0] * x.shape[1], -1, *tail), size, size, ceil_mode=True)
            x = y.reshape(*x.shape[:-k], *y.shape[2:])
            axes = [a for a in axes if a < nd - k]
    for a in axes:
        d = 2 + a
        n = x.shape[d]
        if n == 1:
            continue
        if n % 2:
            x = torch.cat([x, x.narrow(d, n - 1, 1)], dim=d)
        x = x.unflatten(d, (x.shape[d] // 2, 2))
        x = x.mean(d + 1) if mode == "mean" else x.amax(d + 1)
    return x


def upsample2(x: torch.Tensor, axes, sizes, padding: str = "zeros") -> torch.Tensor:
    """Double each spatial axis in `axes` by linear interpolation, then crop it to the matching
    length in `sizes`. Past the grid's edge the edge cell repeats ("zeros" padding) or the grid
    wraps ("circular"). With "zeros" this equals F.interpolate(mode="linear",
    align_corners=False)."""
    for a, n_out in zip(axes, sizes):
        d = 2 + a
        n = x.shape[d]
        if n_out == n:
            continue
        if padding == "circular":
            prev, nxt = torch.roll(x, 1, d), torch.roll(x, -1, d)
        else:
            prev = torch.cat([x.narrow(d, 0, 1), x.narrow(d, 0, n - 1)], dim=d)
            nxt = torch.cat([x.narrow(d, 1, n - 1), x.narrow(d, n - 1, 1)], dim=d)
        even, odd = 0.75 * x + 0.25 * prev, 0.75 * x + 0.25 * nxt
        x = torch.stack([even, odd], dim=d + 1).flatten(d, d + 1).narrow(d, 0, n_out)
    return x


class HierNCA(nca.NCA):
    def __init__(self, ndim: int, channels: int = 16, hidden: int = 128, levels: int = 3,
                 context: int = 16, coarse_hidden: int = 32, pool_axes=None,
                 cond_channels: int = 0, causal_axis: int | None = None,
                 shift_blocks: bool = True, **kw):
        """levels counts the fine grid: 1 is the flat NCA, 3 adds two coarser levels.
        pool_axes are the axes that get coarser (default: every axis except causal_axis).
        context is how many channels each coarse level passes down; coarse_hidden is the
        width of each coarse level's network. shift_blocks moves the blocks by a random
        offset every step (see the module docstring). Other arguments are nca.NCA's."""
        if levels < 1:
            raise ValueError("levels counts the fine grid, so it is at least 1")
        if pool_axes is None:
            pool_axes = [a for a in range(ndim) if a != causal_axis]
        pool_axes = tuple(sorted(set(pool_axes)))
        if any(not 0 <= a < ndim for a in pool_axes):
            raise ValueError(f"pool_axes {pool_axes} are not all axes of a {ndim}-D grid")
        if causal_axis is not None and causal_axis in pool_axes:
            raise ValueError("pooling the one-way axis would let later frames reach earlier ones")
        ctx = context if levels > 1 else 0
        compile_ = kw.get("compile", False)
        super().__init__(ndim, channels, hidden, cond_channels=cond_channels + ctx,
                         causal_axis=causal_axis, **kw)
        # Compiling fuses the many small pooling, perception and interpolation ops, which
        # otherwise take as long as the fine step itself.
        self._context = torch.compile(self.context, dynamic=False) if compile_ else self.context
        self.levels = levels
        self.block = 2 ** (levels - 1)  # the coarsest level's block, in fine cells
        self.pool_axes = pool_axes
        self.shift_blocks = shift_blocks
        self.context_channels = ctx
        self.user_cond_channels = cond_channels
        p = nca.perception_channels(2 * channels, ndim, self.perception)
        self.coarse = nn.ModuleList()  # coarse[l - 1] is level l's network
        for level in range(1, levels):
            top = level == levels - 1
            fc0 = nn.Conv1d(p + (0 if top else ctx), coarse_hidden, 1)
            fc1 = nn.Conv1d(coarse_hidden, ctx, 1, bias=False)
            nn.init.zeros_(fc1.weight)  # say nothing until trained
            self.coarse.append(nn.Sequential(fc0, nn.ReLU(), fc1))

    def context(self, x: torch.Tensor) -> torch.Tensor:
        """What every fine cell reads from the levels above this step: (batch, context, *grid)."""
        mean, top = x, x
        summaries = []
        for _ in range(1, self.levels):
            mean, top = pool2(mean, self.pool_axes, "mean"), pool2(top, self.pool_axes, "max")
            summaries.append(torch.cat([mean, top], dim=1))
        h = None
        for level in range(self.levels - 1, 0, -1):
            s = summaries[level - 1]
            y = nca.perceive(s, self.perception, self.padding, self.causal_axis)
            if h is not None:
                y = torch.cat([y, self._up(h, s)], dim=1)
            b = y.shape[0]
            h = self.coarse[level - 1](y.reshape(b, y.shape[1], -1)).reshape(b, -1, *s.shape[2:])
        return self._up(h, x)

    def _up(self, h: torch.Tensor, finer: torch.Tensor) -> torch.Tensor:
        return upsample2(h, self.pool_axes, [finer.shape[2 + a] for a in self.pool_axes], self.padding)

    def block_offsets(self) -> list[int]:
        """A random offset in [0, block) for each pooled axis."""
        return torch.randint(self.block, (len(self.pool_axes),)).tolist()

    def shifted_context(self, x: torch.Tensor, offsets) -> torch.Tensor:
        """The context with every level's blocks starting `offsets` cells earlier along the
        pooled axes. Circular padding rolls the grid there and back. Zero padding puts
        `offset` empty cells in front and the rest of a block behind (so the shape, and the
        compiled context, never change) and crops them off again."""
        dims = [2 + a for a in self.pool_axes]
        if self.padding == "circular":
            c = self._context(torch.roll(x, list(offsets), dims))
            return torch.roll(c, [-o for o in offsets], dims)
        nd = x.dim() - 2
        pad = [0] * (2 * nd)
        index = [slice(None)] * x.dim()
        for a, o in zip(self.pool_axes, offsets):
            k = 2 * (nd - 1 - a)  # F.pad lists the last axis first
            pad[k], pad[k + 1] = o, self.block - 1 - o
            index[2 + a] = slice(o, o + x.shape[2 + a])
        return self._context(F.pad(x, pad))[tuple(index)]

    def forward(self, x: torch.Tensor, fire_rate: float | None = None,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        if self.levels == 1:
            return super().forward(x, fire_rate, cond)
        if self.shift_blocks:
            extra = self.shifted_context(x, self.block_offsets())
        else:
            extra = self._context(x)
        if self.user_cond_channels:
            extra = torch.cat([cond.expand(x.shape[0], -1, *x.shape[2:]).to(extra.dtype), extra], dim=1)
        return super().forward(x, fire_rate, extra)
