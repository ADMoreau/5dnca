# Higher-dimensional neural cellular automata

Growing Neural Cellular Automata ([Mordvintsev et al., Distill 2020](https://distill.pub/2020/growing-ca)) extended from 2D images to 4D and 5D voxel grids.

The NCA is written once, with the number of spatial dimensions as a parameter. PyTorch stops at `conv3d`, so every neighbourhood operation (Sobel gradients, Laplacian, the alive-cell max-pool) is built from per-axis sliced views of a padded grid. The same code runs in 1D through 5D. A browser viewer shows 3D slices of the results, with sliders that move the slice through the extra dimensions.

**Status:** early research code. The 2D port reproduces the original lizard, and a 4D NCA grows a 4D ball and keeps it stable. Comparing two ideas of a fifth dimension (a spatial 5D lattice, and a tree that branches along the 4th axis) is the next step.

## Results so far

### 2D: reproducing the original

Trained with the original "persistent" recipe (pool of 1,024, batch 8, 64–96 steps, 16 channels, 128 hidden units), 3,000 steps on a laptop CPU:

![Lizard growing from a single cell, steps 0 to 200](docs/images/lizard_growth.png)

Grown from one cell, the lizard keeps its shape for at least 1,000 steps (error stays at 10^-4.5 to 10^-4.6). Training samples never run longer than 96 steps.

### 4D: a ball coloured along the 4th axis

A 12⁴ grid, with a target ball coloured red to blue along `w` so the NCA has to use its 4th axis. Each image lays out every 2D (y, x) slice: rows step through `w` and columns through `z`.

| Target | Grown from one cell, step 150 |
|---|---|
| ![4D target](docs/images/ball4d_target_atlas.png) | ![4D grown](docs/images/ball4d_grown_atlas.png) |

Three of four growths reach an error of 10^-3.9 and hold it to step 600. The NCA also becomes nearly mirror-symmetric in z, y and x, which nothing in the training asked for.

That first 4D model stops growing by sensing the edges of its training grid, so it overgrows a bigger world. It is being retrained with wrap-around edges and damage (regeneration) training; `eval_world.py` checks the result.

### Viewer

![Viewer showing an off-centre w slice of the 4D ball, cut open](docs/images/viewer.png)

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests
```

Train, check and view:

```bash
# 2D lizard (about 3 s/step on a laptop CPU)
.venv/bin/python train_lizard.py --steps 3000

# 4D ball with wrap-around edges and regeneration training
.venv/bin/python train.py --ndim 4 --side 12 --radius 3.5 --padding circular --damage 2 --steps 2000 \
    --out runs/ball4d_wrap

# Several seeds in a bigger world, with damage partway through
.venv/bin/python eval_world.py runs/ball4d_wrap

# Viewer: then open http://localhost:8000/viewer/
.venv/bin/python serve.py
```

Each run writes to `runs/<name>/`:
- `model.pt`: the trained model
- `loss.jsonl`: the loss log
- snapshot images
- `growth_rgba.npy`: every step grown from one seed, as uint8 `(T, *grid, 4)`. The viewer lists these automatically.

## Layout

| File | What it does |
|---|---|
| `nca.py` | The NCA for any number of dimensions: perception (`sobel` or `isotropic`), alive mask, update rule, optional gradient checkpointing |
| `train_lizard.py` | Reproduces the 2D Distill lizard (reference run) |
| `train.py` | Trains on n-D targets (ball, shell) with pool training, optional damage and wrap-around edges |
| `eval_world.py` | Runs a trained model in a bigger world with several seeds; checks size, separation and regrowth |
| `serve.py` | Local server for the viewer (127.0.0.1 only); lists runs at `/api/runs` |
| `viewer/index.html` | three.js viewer: growth-step scrubbing, extra-axis sliders with blending, cut-away, alpha threshold |
| `tests/` | Checks the n-D operators against `conv2d`/`conv3d`/`max_pool` and brute-force 3ⁿ stencils in 4D/5D, plus sparse, dense and checkpointed equivalence |

## Implementation notes

- **Perception without conv4d.** The grid is padded once, and each neighbour is a sliced view. The n-D Sobel along an axis is a central difference on that axis with [1, 2, 1] smoothing on the others, which in 2D is exactly the original kernel. The smoothing passes are shared between axes: 12 one-axis passes in 4D instead of 16.
- **Isotropic perception** (identity, Laplacian, gradient magnitude) gives 3C inputs in any dimension. One set of weights could then run in 2D, 3D and 4D.
- **`skip_dead=True`** runs the per-cell network only on cells that are alive and updating that step. In 4D/5D a shape fills a small part of the grid, so this is much cheaper. It differs from the original rule only in whether a dead cell's update can briefly keep a live neighbour alive, which is why it's off for the reference lizard.
- **`compile=True`** fuses perception with `torch.compile`, about 4× faster in 4D on CPU.
- **Cost** grows as side^n: 32⁴ is about 1M cells. On a laptop CPU a 12⁴ training step takes about 3 seconds, so bigger 4D and 5D runs need a GPU.

## Roadmap

- Bouncing-ball test: a ball's history frozen into the 4th axis, then built two ways:
  - a spatial 5D lattice, with bounciness varying smoothly along `v`
  - a tree that forks along `w` at the first impact
  
  Compare both against independent runs on fit, regrowth and damage leaking between branches.
- Does information flowing backward along `w` reshape the trunk? Compare two-way and parent-to-child links in the tree.
- Live viewer mode: click to plant new seeds or erase regions while the model runs.
- Larger grids (16⁴, 5D) on a GPU.

## Credits

- Mordvintsev, Randazzo, Niklasson and Levin, [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca), Distill 2020.
- Lizard target: `data/lizard_128.png` is the 🦎 emoji from [Noto Emoji](https://github.com/googlefonts/noto-emoji) v2.047, Apache License 2.0.
