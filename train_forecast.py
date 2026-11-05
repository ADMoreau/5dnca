"""Forecast the bounce from the observed fall: K candidate futures, trained with the energy score.

The ball of bounce.py falls, hits the floor and bounces with an unknown bounciness e. The
fall (every frame up to the first impact) and 0..--obs-max frames after it are observed:
their RGBA is clamped to the truth at every step, and the NCA fills in the rest. e is drawn
at random for each training example and is never an input, so with only the fall observed
the right answer is a spread of bounces.

The K candidates are trained together with the energy score, a multi-dimensional CRPS:
each candidate is rewarded for being close to what happened, and the set for being spread
out. The score is proper, so its best expected value comes when the candidates are spread
like the real outcomes.

  lattice      5D NCA on (v, w, z, y, x): the candidates sit side by side along v and see
               each other.
  independent  K separate 4D histories with shared weights; candidates never touch.
  tree         one 4D history whose w axis branches at the first impact (tree_nca.py): the
               fall is a single shared trunk and each candidate is a branch. With --two-way
               the fork reads the mean of its children, so candidates feel each other only
               through the trunk; one-way, they never do.

Every candidate also gets its index as a fixed one-hot input (the shared trunk gets none),
so the modes differ in how the candidates are connected. Time (w) is one-way unless
--two-way: a frame reads only earlier frames.

Training e is uniform on [--e-min, --e-max] minus the held-out band --holdout, on a 0.005
grid. eval.json scores the candidates with only the fall observed (energy score, spread,
the bounciness each candidate looks like, and the best K exact bounces for reference) and
with 1..obs-max frames of the bounce observed, on trained and held-out e.

    .venv/bin/python train_forecast.py --mode lattice --steps 1000 --batch 4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint

import bounce
import nca
from train_bounce import side_view
from tree_nca import TreeNCA

ROOT = Path(__file__).resolve().parent
MODES = ("lattice", "independent", "tree")
E_STEP = 0.005


def energy_score(cands: torch.Tensor, truth: torch.Tensor, eps: float = 1e-8):
    """Energy score of K candidates against what happened, one value per example.

    cands (B, K, ...), truth (B, ...); distances are RMS over all remaining elements.
    ES = mean_k d(c_k, y) - sum_{k,j} d(c_k, c_j) / (2 K^2), the energy score of the K-point
    distribution the candidates form, so it is proper for that distribution. (The "fair"
    version, with K(K-1) in place of K^2, can be gamed by candidates spread very wide.)
    Returns (score (B,), distance to the truth (B, K), pairwise distances (B, K, K))."""
    B, K = cands.shape[:2]
    c = cands.reshape(B, K, -1)
    d_y = ((c - truth.reshape(B, 1, -1)).pow(2).mean(-1) + eps).sqrt()
    d_cc = ((c[:, :, None] - c[:, None]).pow(2).mean(-1) + eps).sqrt()
    d_cc = d_cc * (1 - torch.eye(K, dtype=c.dtype, device=c.device))
    return d_y.mean(1) - d_cc.sum((1, 2)) / (2 * K * K), d_y, d_cc


def spread(d_cc: torch.Tensor) -> torch.Tensor:
    """Mean distance between two different candidates, from pairwise distances (..., K, K)."""
    K = d_cc.shape[-1]
    return d_cc.sum((-2, -1)) / (K * (K - 1))


def best_exact(dist: np.ndarray, atoms: np.ndarray, outcomes: np.ndarray, K: int, rounds: int = 50):
    """The K exact bounces (chosen from `atoms`) with the lowest expected energy score when the
    outcome is uniform over `outcomes`, by coordinate descent. dist is the RMSE between every
    pair of exact bounces. A reference for what K physically correct candidates can achieve."""
    to_out = dist[np.ix_(atoms, outcomes)].mean(1)
    between = dist[np.ix_(atoms, atoms)]

    def score(s):
        return to_out[s].mean() - between[np.ix_(s, s)].sum() / (2 * K * K)

    s = list(np.linspace(0, len(atoms) - 1, K + 2)[1:-1].round().astype(int))
    best = score(s)
    for _ in range(rounds):
        improved = False
        for k in range(K):
            for j in range(len(atoms)):
                t = s.copy()
                t[k] = j
                v = score(t)
                if v < best - 1e-12:
                    s, best, improved = t, v, True
        if not improved:
            break
    return sorted(int(atoms[i]) for i in s), float(best)


class Bounces:
    """Exact histories on a 0.005 grid of bounciness, split into trained and held-out values."""

    def __init__(self, cfg: bounce.BounceConfig, e_min: float, e_max: float, holdout: tuple[float, float]):
        self.e = np.round(np.arange(0, 1 + E_STEP / 2, E_STEP), 3)
        self.hist = torch.from_numpy(np.stack([bounce.render_history(float(e), cfg) for e in self.e]))
        tol = 1e-9
        in_range = (self.e >= e_min - tol) & (self.e <= e_max + tol)
        held = (self.e >= holdout[0] - tol) & (self.e <= holdout[1] + tol)
        self.train = np.nonzero(in_range & ~held)[0]
        self.held = np.nonzero(in_range & held)[0]

    def index(self, e: float) -> int:
        return int(np.argmin(np.abs(self.e - e)))


class Forecast:
    """Model, state layout, candidate-index inputs and the observation clamp for one mode.

    Lattice states are (B, C, K, W, Z, Y, X). Independent states are (B*K, C, W, Z, Y, X),
    example by example. Tree states are (B, C, N, Z, Y, X): the fork trunk nodes, then each
    candidate's W - fork branch nodes. cands() views any of them as (B, K, C, W, Z, Y, X)
    (in the tree, every candidate's history starts with the same trunk); per_example() is
    the layout the pool stores."""

    def __init__(self, mode: str, cfg: bounce.BounceConfig, args):
        self.mode, self.K = mode, args.candidates
        self.fork = bounce.first_impact_index(cfg)
        causal = not args.two_way
        kw = dict(channels=args.channels, hidden=args.hidden, perception=args.perception,
                  padding="zeros", skip_dead=True, compile=not args.no_compile, cond_channels=self.K)
        onehot = torch.eye(self.K)
        if mode == "lattice":
            self.model = nca.NCA(5, causal_axis=1 if causal else None, **kw)
            self.grid = (self.K, cfg.W, cfg.Z, cfg.Y, cfg.X)
            self._cond = onehot[None, :, :, None, None, None, None]   # channel k is 1 at v = k
        elif mode == "independent":
            self.model = nca.NCA(4, causal_axis=0 if causal else None, **kw)
            self.grid = (cfg.W, cfg.Z, cfg.Y, cfg.X)
            self._cond = onehot[:, :, None, None, None, None]          # row k is candidate k
        elif mode == "tree":
            wf = cfg.W - self.fork
            parent, node_w, branch = list(range(-1, self.fork - 1)), list(range(self.fork)), [-1] * self.fork
            for k in range(self.K):
                first = len(parent)
                parent += [self.fork - 1] + list(range(first, first + wf - 1))
                node_w += list(range(self.fork, cfg.W))
                branch += [k] * wf
            self.node_w = torch.tensor(node_w)
            self.model = TreeNCA(parent, edges="down" if causal else "both", **kw)
            self.grid = (len(parent), cfg.Z, cfg.Y, cfg.X)
            branch = torch.tensor(branch)
            on = torch.nonzero(branch >= 0).squeeze(1)
            labels = torch.zeros(self.K, len(parent))
            labels[branch[on], on] = 1.0                                 # the trunk has no label
            self._cond = labels[None, :, :, None, None, None]
            # Node indices of each candidate's history: the trunk, then its own branch. (K, W)
            self._paths = torch.cat([torch.arange(self.fork).expand(self.K, -1),
                                     self.fork + torch.arange(self.K)[:, None] * wf + torch.arange(wf)], dim=1)
        else:
            raise ValueError(mode)

    def rows(self, B: int) -> int:
        return B * self.K if self.mode == "independent" else B

    def cond(self, B: int) -> torch.Tensor:
        return self._cond.repeat(B, 1, 1, 1, 1, 1) if self.mode == "independent" else self._cond

    def cands(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "lattice":
            return x.movedim(2, 1)
        if self.mode == "tree":
            return x.index_select(2, self._paths.flatten()).unflatten(2, tuple(self._paths.shape)).movedim(2, 1)
        return x.reshape(-1, self.K, *x.shape[1:])

    def per_example(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, self.K, *x.shape[1:]) if self.mode == "independent" else x

    def from_examples(self, p: torch.Tensor) -> torch.Tensor:
        return p.reshape(-1, *p.shape[2:]) if self.mode == "independent" else p

    def observation(self, truth: torch.Tensor, n_obs: torch.Tensor):
        """The true RGBA and a mask of the observed frames (everything before the first impact,
        plus n_obs frames after it), in the model's layout. truth (B, 4, W, Z, Y, X)."""
        seen = torch.arange(truth.shape[2])[None] < (self.fork + n_obs)[:, None]   # (B, W)
        if self.mode == "lattice":
            return truth[:, :, None], seen[:, None, None, :, None, None, None]
        if self.mode == "tree":
            return truth[:, :, self.node_w], seen[:, self.node_w][:, None, :, None, None, None]
        return (truth.repeat_interleave(self.K, 0),
                seen.repeat_interleave(self.K, 0)[:, None, :, None, None, None])

    @staticmethod
    def clamp(x: torch.Tensor, obs) -> torch.Tensor:
        """Reset the RGBA of observed frames to the truth; hidden channels stay the model's own."""
        t, m = obs
        return torch.cat([torch.where(m, t, x[:, :4]), x[:, 4:]], dim=1)

    def start(self, obs, B: int) -> torch.Tensor:
        """Observed frames filled in, the future empty."""
        return self.clamp(torch.zeros(self.rows(B), self.model.channels, *self.grid), obs)

    def rollout(self, x, steps: int, obs, cond, checkpoint: bool = False) -> torch.Tensor:
        for _ in range(steps):
            if checkpoint and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(self.model, x, None, cond, use_reentrant=False)
            else:
                x = self.model(x, cond=cond)
            x = self.clamp(x, obs)
        return x

    def future(self, x: torch.Tensor) -> torch.Tensor:
        """RGBA after the first impact, per candidate: (B, K, 4, W - fork, Z, Y, X)."""
        return self.cands(x)[:, :, :4, self.fork:]


def evaluate(fc: Forecast, book: Bounces, args, out: Path) -> dict:
    """Score forecasts on exact bounces. Distances are RMS over the RGBA of the frames after
    the first impact, computed in float64."""
    K, fork, model = fc.K, fc.fork, fc.model
    flat = book.hist[:, :, fork:].reshape(len(book.e), -1).double()
    scale = flat.shape[1] ** 0.5

    def to_book(c):                      # (K, 4, Wf, Z, Y, X) -> RMSE to every exact bounce (K, M)
        return torch.cdist(c.reshape(len(c), -1).double(), flat) / scale

    def between(c):                      # (K, ...) -> RMSE between candidates (K, K)
        f = c.reshape(len(c), -1).double()
        return torch.cdist(f, f) / scale

    oracle, oracle_es = best_exact((torch.cdist(flat, flat) / scale).numpy(), book.train, book.train, K)
    single, single_es = best_exact((torch.cdist(flat, flat) / scale).numpy(), book.train, book.train, 1)
    e_test = np.round(np.arange(args.e_min, args.e_max + 1e-9, 0.05), 3)
    test_idx = [book.index(e) for e in e_test]
    held = set(book.held.tolist())
    show = book.index(float(np.mean(args.holdout)))       # a held-out bounce to draw

    with torch.no_grad():
        # Only the fall observed: the input is the same for every e, so one forecast serves all.
        obs = fc.observation(book.hist[book.train[:1]], torch.zeros(1, dtype=torch.long))
        cond = fc.cond(1)
        x = fc.start(obs, 1)

        def rgba_u8(s):
            return fc.cands(s)[0, :, :4].clamp(0, 1).mul(255).round().byte().movedim(1, -1)

        frames = [rgba_u8(x)]
        for t in range(1, args.eval_steps + 1):
            x = fc.clamp(model(x, cond=cond), obs)
            if t % 2 == 0:
                frames.append(rgba_u8(x))
        c0 = fc.future(x)[0]
        D, P0 = to_book(c0), between(c0)
        es_by_e = D.mean(0) - P0.sum() / (2 * K * K)
        x_long = fc.rollout(x, 2 * args.eval_steps, obs, cond)
        c_long = fc.future(x_long)[0]
        es_long = (to_book(c_long).mean(0) - between(c_long).sum() / (2 * K * K))[book.train].mean()
        side_view(list(fc.cands(x)[0]) + [book.hist[i] for i in oracle]).save(out / "fall_only.png")
        np.save(out / "growth_rgba.npy", torch.stack(frames).numpy())          # (T, K, W, Z, Y, X, 4)

        spread0 = float(P0.sum() / (K * (K - 1)))
        per_e = {"0": [{"e": float(book.e[i]), "held_out": i in held, "es": float(es_by_e[i]),
                        "best_rmse": float(D[:, i].min()), "spread": spread0} for i in test_idx]}
        for n_obs in range(1, args.obs_max + 1):
            rows = []
            for s in range(0, len(test_idx), 8):
                chunk = test_idx[s:s + 8]
                truth = book.hist[chunk]
                obs = fc.observation(truth, torch.full((len(chunk),), n_obs))
                xs = fc.rollout(fc.start(obs, len(chunk)), args.eval_steps, obs, fc.cond(len(chunk)))
                for j, i in enumerate(chunk):
                    c = fc.future(xs)[j]
                    Dj, Pj = to_book(c), between(c)
                    near = int(Dj[:, i].argmin())
                    rows.append({"e": float(book.e[i]), "held_out": i in held,
                                 "es": float(Dj[:, i].mean() - Pj.sum() / (2 * K * K)),
                                 "best_rmse": float(Dj[near, i]), "spread": float(Pj.sum() / (K * (K - 1))),
                                 "candidate_e": [float(book.e[k]) for k in Dj.argmin(1)],
                                 "best_e_error": abs(float(book.e[int(Dj[near].argmin())] - book.e[i]))})
                    if n_obs == 1 and i == show:
                        side_view(list(fc.cands(xs)[j]) + [truth[j]]).save(out / f"observed1_e{book.e[i]:.2f}.png")
            per_e[str(n_obs)] = rows

    def summary(rows):
        out_ = {}
        for name, keep in (("trained", False), ("held_out", True)):
            sel = [r for r in rows if r["held_out"] == keep]
            out_[name] = {k: round(float(np.mean([r[k] for r in sel])), 5)
                          for k in ("es", "best_rmse", "spread", "best_e_error") if k in sel[0]}
        return out_

    res = {
        "fall_only": {
            "candidate_e": [float(book.e[k]) for k in D.argmin(1)],
            "candidate_match_rmse": [round(float(m), 5) for m in D.min(1).values],
            "expected_es_trained": round(float(es_by_e[book.train].mean()), 5),
            "expected_es_held_out": round(float(es_by_e[book.held].mean()), 5),
            f"expected_es_trained_t{3 * args.eval_steps}": round(float(es_long), 5),
            "spread": round(spread0, 5),
            "best_exact_e": [float(book.e[i]) for i in oracle], "best_exact_es": round(oracle_es, 5),
            "single_exact_e": float(book.e[single[0]]), "single_exact_es": round(single_es, 5),
        },
        "by_frames_observed": {n: summary(rows) for n, rows in per_e.items()},
        "eval_steps": args.eval_steps,
        "per_e": per_e,
    }
    (out / "eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("fall_only", "by_frames_observed")}), flush=True)
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--candidates", type=int, default=4)
    p.add_argument("--two-way", action="store_true", help="let frames read later frames too")
    p.add_argument("--obs-max", type=int, default=2, help="most frames after the impact that can be observed")
    p.add_argument("--e-min", type=float, default=0.2)
    p.add_argument("--e-max", type=float, default=0.95)
    p.add_argument("--holdout", type=float, nargs=2, default=(0.55, 0.70), metavar=("LO", "HI"))
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--min-iters", type=int, default=64)
    p.add_argument("--max-iters", type=int, default=96)
    p.add_argument("--lr-drop", type=int, default=None, help="default: steps*2/3")
    p.add_argument("--channels", type=int, default=16)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--perception", choices=["sobel", "isotropic"], default="sobel")
    p.add_argument("--batch", type=int, default=4, help="examples per step, each with K candidates")
    p.add_argument("--pool", type=int, default=256)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-checkpoint", action="store_true", help="keep every step's activations (more memory)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--snap-every", type=int, default=100)
    p.add_argument("--eval-steps", type=int, default=128, help="steps each evaluation forecast runs")
    p.add_argument("--bench", type=int, default=0, help="time this many training steps and exit")
    p.add_argument("--eval-only", action="store_true", help="evaluate the model saved in --out")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = bounce.BounceConfig()
    book = Bounces(cfg, args.e_min, args.e_max, tuple(args.holdout))
    fc = Forecast(args.mode, cfg, args)
    model = fc.model
    out = args.out or ROOT / "runs" / f"forecast_{args.mode}"
    if args.eval_only:
        model.load_state_dict(torch.load(out / "model.pt")["model"])
        evaluate(fc, book, args, out)
        return
    if not args.bench:
        out.mkdir(parents=True, exist_ok=True)

    P, B = args.pool, args.batch

    def draw(n):
        return (torch.from_numpy(rng.choice(book.train, n)),
                torch.from_numpy(rng.integers(0, args.obs_max + 1, n)))

    def fresh(e_idx, n_obs):
        return fc.per_example(fc.start(fc.observation(book.hist[e_idx], n_obs), len(e_idx)))

    pool_e, pool_n = draw(P)
    pool = torch.empty(P, *fresh(pool_e[:1], pool_n[:1]).shape[1:])
    for i in range(0, P, 32):
        pool[i:i + 32] = fresh(pool_e[i:i + 32], pool_n[i:i + 32])

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    lr_drop = args.lr_drop or (2 * args.steps) // 3
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_drop], gamma=0.1)
    config = vars(args) | {"out": str(out), "lr_drop": lr_drop, "bounce": cfg.__dict__, "grid": fc.grid,
                           "fork_w": fc.fork, "n_trained_e": len(book.train), "n_held_out_e": len(book.held)}
    config = json.loads(json.dumps(config, default=str))
    print(config, flush=True)
    if not args.bench:
        (out / "config.json").write_text(json.dumps(config, indent=2))
    log = None if args.bench else open(out / "loss.jsonl", "a")
    cond = fc.cond(B)

    t0 = time.time()
    for step in range(1, (args.bench or args.steps) + 1):
        idx = torch.from_numpy(rng.choice(P, B, replace=False))
        e_i, n_i, x0 = pool_e[idx], pool_n[idx], pool[idx]
        with torch.no_grad():
            es0 = energy_score(fc.future(fc.from_examples(x0)), book.hist[e_i][:, :, fc.fork:])[0]
        # The worst forecast in the batch is replaced by a new example: new e, new observation.
        worst = int(torch.argmax(es0))
        e_new, n_new = draw(1)
        e_i[worst], n_i[worst] = e_new[0], n_new[0]
        x0[worst] = fresh(e_new, n_new)[0]
        truth = book.hist[e_i]
        obs = fc.observation(truth, n_i)

        iters = int(rng.integers(args.min_iters, args.max_iters + 1))
        x = fc.rollout(fc.from_examples(x0), iters, obs, cond, checkpoint=not args.no_checkpoint)
        es, d_y, d_cc = energy_score(fc.future(x), truth[:, :, fc.fork:])
        loss = es.mean()

        opt.zero_grad()
        loss.backward()
        with torch.no_grad():
            for prm in model.parameters():
                prm.grad /= prm.grad.norm() + 1e-8
        opt.step()
        sched.step()
        pool[idx] = fc.per_example(x.detach())
        pool_e[idx], pool_n[idx] = e_i, n_i

        rec = {"step": step, "loss": loss.item(), "log10": round(float(np.log10(loss.item())), 3),
               "best": round(d_y.detach().min(1).values.mean().item(), 5),
               "spread": round(spread(d_cc.detach()).mean().item(), 5), "sec": round(time.time() - t0, 1)}
        if args.bench:
            print(rec | {"iters": iters}, flush=True)
            continue
        if step % args.log_every == 0 or step == 1:
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if step % args.snap_every == 0 or step == args.steps:
            side_view(list(fc.cands(x.detach())[0]) + [truth[0]]).save(out / f"snap_{step:05d}.png")
            torch.save({"model": model.state_dict(), "config": config, "step": step}, out / "model.pt")
    if args.bench:
        return
    evaluate(fc, book, args, out)


if __name__ == "__main__":
    main()
