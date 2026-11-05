"""Neural cellular automaton on an n-dimensional grid.

A port of Growing Neural Cellular Automata (Mordvintsev et al., Distill 2020)
in which the number of spatial dimensions is a parameter. PyTorch has no
conv4d, so every neighbourhood operation (Sobel gradients, Laplacian, alive
max-pool) is built from per-axis shifts. That costs O(n) shifts per channel
instead of a dense 3^n kernel, and it is the same code for n = 1..5.

State layout is channel-first, like conv2d/conv3d: (batch, channels, *spatial).
Channels 0-3 are premultiplied RGBA; channel 3 (alpha) marks living cells.

One axis can be made one-way (causal_axis), for when it is time: along it a cell reads
its predecessor and itself, never its successor, so information only flows forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

ALPHA = 3


def pad1(x: torch.Tensor, padding: str = "zeros", fill: float = 0.0) -> torch.Tensor:
    """Pad every spatial axis by one cell: with `fill` ("zeros") or by wrapping ("circular")."""
    nd = x.dim() - 2
    if padding == "zeros":
        return F.pad(x, (1, 1) * nd, value=fill)
    if padding == "circular":
        for a in range(nd):
            d = 2 + a
            x = torch.cat([x.narrow(d, -1, 1), x, x.narrow(d, 0, 1)], dim=d)
        return x
    raise ValueError(f"unknown padding {padding!r}")


def pad_axes(x: torch.Tensor, axes, padding: str = "zeros", fill: float = 0.0) -> torch.Tensor:
    """pad1 restricted to some spatial axes; the others are left alone."""
    nd = x.dim() - 2
    axes = set(axes)
    if padding == "zeros":
        pads = []
        for a in reversed(range(nd)):
            pads += [1, 1] if a in axes else [0, 0]
        return F.pad(x, tuple(pads), value=fill)
    if padding == "circular":
        for a in sorted(axes):
            d = 2 + a
            x = torch.cat([x.narrow(d, -1, 1), x, x.narrow(d, 0, 1)], dim=d)
        return x
    raise ValueError(f"unknown padding {padding!r}")


def shift_prev(x: torch.Tensor, axis: int, fill: float = 0.0) -> torch.Tensor:
    """Each cell's predecessor along a spatial axis (x[i - 1]), with `fill` before the first
    cell. Never wraps: a one-way axis is time, and time does not loop."""
    d = 2 + axis
    first = torch.full_like(x.narrow(d, 0, 1), fill)
    return torch.cat([first, x.narrow(d, 0, x.shape[d] - 1)], dim=d)


def _triple(x: torch.Tensor, axis: int):
    """Left, centre and right neighbours along an axis, as views. Shrinks that axis by 2."""
    d = 2 + axis
    n = x.shape[d] - 2
    return x.narrow(d, 0, n), x.narrow(d, 1, n), x.narrow(d, 2, n)


def _smooth(x: torch.Tensor, axis: int) -> torch.Tensor:
    """[1, 2, 1] / 4 along one (padded) axis."""
    lft, mid, rgt = _triple(x, axis)
    return (lft + rgt + 2 * mid) * 0.25


def _diff(x: torch.Tensor, axis: int) -> torch.Tensor:
    """Central difference [-1, 0, 1] / 2 along one (padded) axis, cross-correlation order like conv."""
    lft, _, rgt = _triple(x, axis)
    return (rgt - lft) * 0.5


def _sobels(xp: torch.Tensor, axes: list[int]) -> dict[int, torch.Tensor]:
    """Sobel along each axis in `axes` of a tensor padded on those axes (and only those
    still padded). Splits the axes in half so the smoothing passes are shared:
    12 one-axis passes in 4D instead of 16, 17 in 5D instead of 25."""
    if len(axes) == 1:
        return {axes[0]: _diff(xp, axes[0])}
    half = len(axes) // 2
    first, second = axes[:half], axes[half:]
    out = {}
    for keep, smooth in ((first, second), (second, first)):
        t = xp
        for b in smooth:
            t = _smooth(t, b)
        out |= _sobels(t, keep)
    return out


def _causal_sobels(x: torch.Tensor, axis: int, padding: str) -> dict[int, torch.Tensor]:
    """Sobels when `axis` is one-way: along it the successor is replaced by the cell itself,
    so the derivative along it is (self - prev) / 2 and the [1, 2, 1] smoothing along it
    becomes (prev + 3 self) / 4. On a chain this is tree_nca.TreeNCA(edges="down")."""
    nd = x.dim() - 2
    others = [a for a in range(nd) if a != axis]
    xp = pad_axes(x, others, padding)
    prev = shift_prev(xp, axis)
    g = (xp - prev) * 0.5
    for a in others:
        g = _smooth(g, a)
    out = {axis: g}
    if others:
        out |= _sobels((prev + 3 * xp) * 0.25, others)
    return out


def sobels(x: torch.Tensor, padding: str = "zeros", causal_axis: int | None = None) -> list[torch.Tensor]:
    """n-D Sobel derivative along every axis: difference on that axis, [1, 2, 1] smoothing
    on all others. In 2D this is exactly the original's outer([1, 2, 1], [-1, 0, 1]) / 8."""
    nd = x.dim() - 2
    if causal_axis is None:
        out = _sobels(pad1(x, padding), list(range(nd)))
    else:
        out = _causal_sobels(x, causal_axis, padding)
    return [out[a] for a in range(nd)]


def sobel(x: torch.Tensor, axis: int, padding: str = "zeros") -> torch.Tensor:
    return sobels(x, padding)[axis]


def laplacian(x: torch.Tensor, padding: str = "zeros", causal_axis: int | None = None) -> torch.Tensor:
    """Sum of per-axis second differences (the (2n+1)-point stencil). Along a one-way axis
    the successor is replaced by the cell itself, which leaves prev - self."""
    nd = x.dim() - 2
    xp = pad1(x, padding)
    out = -2 * nd * x
    for a in range(nd):
        if a == causal_axis:
            out = out + shift_prev(x, a) + x
            continue
        t = xp
        for b in range(nd):
            if b != a:
                t = t.narrow(2 + b, 1, xp.shape[2 + b] - 2)
        lft, _, rgt = _triple(t, a)
        out = out + lft + rgt
    return out


def max_pool3(x: torch.Tensor, padding: str = "zeros", causal_axis: int | None = None) -> torch.Tensor:
    """3^n max-pool with stride 1, done separably. Out-of-grid cells are ignored. Along a
    one-way axis only the predecessor and the cell itself count."""
    nd = x.dim() - 2
    axes = [a for a in range(nd) if a != causal_axis]
    out = pad_axes(x, axes, padding, fill=float("-inf"))
    for a in axes:
        lft, mid, rgt = _triple(out, a)
        out = torch.maximum(torch.maximum(lft, mid), rgt)
    if causal_axis is not None:
        out = torch.maximum(out, shift_prev(out, causal_axis, fill=float("-inf")))
    return out


def perceive(x: torch.Tensor, mode: str = "sobel", padding: str = "zeros",
             causal_axis: int | None = None) -> torch.Tensor:
    """Per-cell perception vector.

    "sobel":     identity + one Sobel per axis   -> C * (n + 1) channels (the original).
    "isotropic": identity, Laplacian, |gradient| -> 3C channels in any n, so the
                 same weights can run at any dimension (research framing 1).

    Every neighbourhood op pads once and reads neighbours as sliced views, so the
    same code serves any n without conv4d/conv5d.
    """
    if mode == "sobel":
        return torch.cat([x] + sobels(x, padding, causal_axis), dim=1)
    if mode == "isotropic":
        grad_sq = sum(g * g for g in sobels(x, padding, causal_axis))
        return torch.cat([x, laplacian(x, padding, causal_axis), torch.sqrt(grad_sq + 1e-8)], dim=1)
    raise ValueError(f"unknown perception mode {mode!r}")


def perception_channels(channels: int, ndim: int, mode: str) -> int:
    return channels * (ndim + 1) if mode == "sobel" else channels * 3


def alive_mask(x: torch.Tensor, padding: str = "zeros", threshold: float = 0.1,
               causal_axis: int | None = None) -> torch.Tensor:
    # A boolean mask carries no gradient, so don't record the max-pool for backprop.
    with torch.no_grad():
        return max_pool3(x[:, ALPHA : ALPHA + 1], padding, causal_axis) > threshold


class NCA(nn.Module):
    def __init__(
        self,
        ndim: int,
        channels: int = 16,
        hidden: int = 128,
        fire_rate: float = 0.5,
        perception: str = "sobel",
        padding: str = "zeros",
        skip_dead: bool = False,
        compile: bool = False,
        cond_channels: int = 0,
        causal_axis: int | None = None,
    ):
        """skip_dead=False is the original rule: every cell computes an update, and the
        alive mask is applied afterwards. skip_dead=True only updates cells that are
        alive before the step (and fire), and runs the network on just those cells.
        That is much cheaper when the shape fills a small part of the grid, as in 4D/5D
        (an inscribed ball fills 31% of a 4D cube, 16% of a 5D one). The two rules differ
        only in whether a dead cell's update can briefly keep a live neighbour alive.

        compile=True runs perception through torch.compile, which fuses the per-axis
        neighbour reads into one loop (about 4x faster in 4D on CPU, identical output).

        cond_channels > 0 adds fixed per-cell inputs (passed to forward as `cond`) that
        the network reads but never changes, e.g. which history a cell belongs to.

        causal_axis makes that spatial axis one-way, for when it is time: in perception and
        in the alive mask a cell reads its predecessor along it and itself, never its
        successor. The axis is never wrapped, whatever the padding."""
        super().__init__()
        if causal_axis is not None and not 0 <= causal_axis < ndim:
            raise ValueError(f"causal_axis {causal_axis} is not an axis of a {ndim}-D grid")
        self.causal_axis = causal_axis
        self.ndim = ndim
        self.channels = channels
        self.fire_rate = fire_rate
        self.perception = perception
        self.padding = padding
        self.skip_dead = skip_dead
        self._perceive = torch.compile(perceive, dynamic=False) if compile else perceive
        self.cond_channels = cond_channels
        p = perception_channels(channels, ndim, perception) + cond_channels
        # 1x1 convolutions, applied to the flattened grid so they work for any n.
        self.fc0 = nn.Conv1d(p, hidden, 1)
        self.fc1 = nn.Conv1d(hidden, channels, 1, bias=False)
        nn.init.zeros_(self.fc1.weight)  # start as the "do nothing" rule

    def seed(self, shape: tuple[int, ...], batch: int = 1, device=None) -> torch.Tensor:
        """One live cell at the centre of a grid of the given spatial shape."""
        assert len(shape) == self.ndim
        x = torch.zeros(batch, self.channels, *shape, device=device)
        x[(slice(None), slice(ALPHA, None)) + tuple(s // 2 for s in shape)] = 1.0
        return x

    def perceive_cells(self, x: torch.Tensor) -> torch.Tensor:
        """Perception vector of every cell. Subclasses with other neighbourhoods override this."""
        return self._perceive(x, self.perception, self.padding, self.causal_axis)

    def alive(self, x: torch.Tensor) -> torch.Tensor:
        """Cells with a live neighbour (alpha > 0.1 within the 3^n neighbourhood)."""
        return alive_mask(x, self.padding, causal_axis=self.causal_axis)

    def forward(self, x: torch.Tensor, fire_rate: float | None = None,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        """One CA step. `cond` (batch, cond_channels, ...) broadcasts over the grid."""
        assert x.dim() == self.ndim + 2, f"expected {self.ndim}-D grid, got shape {tuple(x.shape)}"
        pre_alive = self.alive(x)

        y = self.perceive_cells(x)
        b = y.shape[0]
        spatial = y.shape[2:]
        rate = self.fire_rate if fire_rate is None else fire_rate
        fire = torch.rand(b, 1, *spatial, device=x.device) <= rate

        if self.skip_dead:
            # Gather the cells that will change straight from the channel-first layout,
            # run the MLP on just those, and scatter the update back. The perception and
            # cond are gathered separately, so the full grid is never copied to join them.
            bi, ni = (fire & pre_alive).reshape(b, -1).nonzero(as_tuple=True)
            cells = y.reshape(b, y.shape[1], -1)[bi, :, ni]  # (active, perception)
            if self.cond_channels:
                c = cond.expand(b, -1, *spatial).reshape(b, self.cond_channels, -1)[bi, :, ni]
                cells = torch.cat([cells, c.to(cells.dtype)], dim=1)
            h = F.relu(F.linear(cells, self.fc0.weight.squeeze(-1), self.fc0.bias))
            d = F.linear(h, self.fc1.weight.squeeze(-1))
            dx = x.new_zeros(b, self.channels, y[0, 0].numel())
            dx[bi, :, ni] = d  # autograd records this as a scatter into the zero tensor
            x = x + dx.reshape(x.shape)
        else:
            if self.cond_channels:
                y = torch.cat([y, cond.expand(b, -1, *spatial).to(y.dtype)], dim=1)
            dx = self.fc1(F.relu(self.fc0(y.reshape(b, y.shape[1], -1)))).reshape(b, self.channels, *spatial)
            x = x + dx * fire.to(x.dtype)

        post_alive = self.alive(x)
        return x * (pre_alive & post_alive).to(x.dtype)

    def rollout(self, x: torch.Tensor, steps: int, checkpoint: bool = False,
                cond: torch.Tensor | None = None) -> torch.Tensor:
        """Run `steps` CA steps. With checkpoint=True only the state between steps is kept
        for backprop and each step is recomputed in the backward pass (same random update
        masks, since torch.utils.checkpoint restores the RNG state). Memory then scales with
        steps * state size instead of steps * (state + perception + hidden activations)."""
        for _ in range(steps):
            if checkpoint and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(self, x, None, cond, use_reentrant=False)
            else:
                x = self(x, cond=cond)
        return x


def to_rgba(x: torch.Tensor) -> torch.Tensor:
    return x[:, :4]


def to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Composite premultiplied RGBA over white."""
    rgb, a = x[:, :3], x[:, 3:4].clamp(0, 1)
    return (1.0 - a + rgb).clamp(0, 1)
