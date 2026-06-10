# wavesimpro-starter-kit

A lightweight companion repository for the [`wavesimpro`](https://pypi.org/project/wavesimpro/) API.

It contains **only** what a user of `wavesimpro` needs to get going:

- **`wsutils/`** — small, *engine-free* helper utilities to build inputs and inspect outputs:
  - `create_medium` — `random_permittivity`, `cuboids_permittivity`, `sphere_permittivity`
  - `create_source` — `point_source`, `plane_wave`, `gaussian_beam`
  - `plotting` — `plot_computed`, `plot_computed_and_reference`
  - `utilities` — `relative_error`, `all_close`, `analytical_solution`, `normalize`, …
- **`examples/`** — runnable scripts that solve the Helmholtz equation and Maxwell's equations
  via `from wavesimpro import simulate`.

These utilities produce plain NumPy arrays and depend only on NumPy/SciPy/Matplotlib — they do
**not** import the `wavesim` core engine. The only wave-simulation dependency is `wavesimpro`.

## Install

The only wave-simulation dependency, [`wavesimpro`](https://pypi.org/project/wavesimpro/), is a
PyPI-only package — it is **not** vendored here. Installing this repo (any path below) pulls it in.
Run the commands from the repository root.

### Option A — conda (recommended)

The conda-friendly scientific stack comes from `conda-forge`; `wavesimpro` is installed via `pip`
(it is not on conda-forge):

```bash
conda env create -f environment.yml
conda activate wavesimpro-starter-kit
pip install -e . --no-deps   # puts the local `wsutils` package on the path
```

`--no-deps` keeps the conda-managed `numpy`/`scipy`/`matplotlib` binaries already in the env.

### Option B — pip / venv

```bash
pip install -r requirements.txt
pip install -e .             # installs the local `wsutils` package
```

(`pip install -e .` alone also works — it resolves the same deps from `pyproject.toml`.)

> **Other tools:** because all dependencies are declared in `pyproject.toml`, tools such as
> [uv](https://docs.astral.sh/uv/) or [Poetry](https://python-poetry.org/) work too — e.g.
> `uv pip install -e .`.

Either way you end up with the `wsutils` package (and `wavesimpro`, `numpy`, `scipy`, `matplotlib`)
on the path, so the example scripts run from anywhere with no `sys.path` juggling.

> **GPU note:** some examples pass `simulate(..., use_gpu=True)`. This is a *server-side* flag
> for the `wavesimpro` cloud platform — nothing runs on your local GPU, and results return as
> plain host NumPy arrays. There is **no** local `cupy` (or other GPU) dependency to install.

## Run an example

```bash
python examples/helmholtz_simulate.py
```

## Layout

```
wavesimpro-starter-kit/
├── pyproject.toml
├── environment.yml   # conda env (conda-forge stack + wavesimpro via pip)
├── requirements.txt  # pip-only install path
├── README.md
├── wsutils/          # engine-free helper utilities
│   ├── create_medium.py
│   ├── create_source.py
│   ├── plotting.py
│   └── utilities.py
└── examples/         # runnable simulate-based scripts
```

## Notes

- `wsutils` was extracted from `wavesim.utilities`. Engine-only helpers that were **not** needed
  by these examples (`add_absorbing_boundaries`, `full_matrix`) were dropped, and the few engine
  type-hints were replaced with plain NumPy types. `all_close` accepts either a NumPy array or any
  object exposing a `.gather()` method (e.g. a wavesim engine array), via duck typing.

## License

[MIT](LICENSE). The `wsutils` helpers are adapted from the MIT-licensed
[`wavesim_py`](https://github.com/IvoVellekoop/wavesim_py) project (Ivo Vellekoop, Swapnil Mache —
University of Twente); the `LICENSE` file retains that original copyright notice alongside
Rayfos Ltd.
