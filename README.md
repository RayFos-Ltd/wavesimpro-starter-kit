# wavesimpro-starter-kit

A lightweight companion repository for **WaveSimPro**, the cloud wave-simulation service by
[Rayfos](https://www.rayfos.com/wavesimpro). The examples here call `simulate` against the
WaveSimPro platform — the [`wavesimpro`](https://pypi.org/project/wavesimpro/) PyPI package is the
thin client that talks to it.

> ### 🔑 You need a WaveSimPro account to run these examples
> The simulations run on the WaveSimPro cloud, not on your machine.
> - **Register / sign in:** **[studio.rayfos.com](https://studio.rayfos.com/)**
> - **Product overview:** **[rayfos.com/wavesimpro](https://www.rayfos.com/wavesimpro)**

> ### 👁️ Scripting power, plus the WaveSimPro Cloud Web GUI to see it
> This kit gives you the best of both: the **freedom and power of scripting** to build and
> submit simulations from Python (your own code, or the `examples/`), *and* the
> **interactive WaveSimPro Cloud Web GUI** to explore what comes back. What's easy to miss
> is that **every simulation you submit — its model and its results — is also waiting for you
> in the Web GUI at [studio.rayfos.com](https://studio.rayfos.com/)**, where you can browse,
> visualise, and inspect it interactively. It's by far the easiest way to explore a 3-D field
> or compare runs — don't rely only on the plots the scripts save locally.
>
> *(You can also build and run simulations entirely in the Web GUI, without scripting — but
> that's outside the scope of this kit.)*

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

## Scientific foundation

WaveSimPro is built on the **modified Born series (MBS)** method for solving the wave equation —
an approach with no numerical dispersion, lower memory requirements, and faster convergence than
the finite-difference time-domain (FDTD) methods traditionally used for these problems. The method
and its open-source reference implementation were developed by **[Ivo Vellekoop](https://github.com/IvoVellekoop)**
and colleagues at the **University of Twente**, to whom this project owes its scientific foundation.

The two publications behind the method:

> Osnabrugge, G., Leedumrongwatthanakun, S., & Vellekoop, I. M. (2016). *A convergent Born series
> for solving the inhomogeneous Helmholtz equation in arbitrarily large media.* Journal of
> Computational Physics, 322, 113–124. <https://doi.org/10.1016/j.jcp.2016.06.034>

> Mache, S., & Vellekoop, I. M. (2025). *Domain decomposition of the modified Born series approach
> for large-scale wave propagation simulations.* Journal of Computational Physics, 114619.
> <https://doi.org/10.1016/j.jcp.2025.114619>

The reference open-source Python implementation, **`wavesim_py`**, is available on GitHub at
**<https://github.com/IvoVellekoop/wavesim_py>** (documentation:
[wavesim.readthedocs.io](https://wavesim.readthedocs.io/); project site:
[wavesim.org](https://www.wavesim.org/)). The `wsutils` helpers in this starter kit are adapted
from it — see [License](#license).

## Install

The only wave-simulation dependency, [`wavesimpro`](https://pypi.org/project/wavesimpro/), is a
PyPI-only package — it is **not** vendored here. Installing this repo (any path below) pulls it in.
Run the commands from the repository root.

### Option A — conda (recommended)

The conda-friendly scientific stack comes from `conda-forge`; `wavesimpro` is installed via `pip`
(it is not on conda-forge).

**Don't have conda yet?** We recommend [Miniconda](https://docs.anaconda.com/miniconda/) (a much
lighter counterpart of Anaconda):

1. **Download Miniconda**, choosing the [installer](https://docs.anaconda.com/miniconda/) for your
   operating system (Windows/macOS/Linux).
2. **Install it**, following the [installation instructions](https://docs.anaconda.com/miniconda/miniconda-install/)
   for your OS. If you are unsure about any setting, accept the defaults. (If you cannot immediately
   use conda, close and re-open your terminal — on Windows, use the **Anaconda Prompt**.)
3. **Test the installation** by running `conda list` — a list of installed packages should appear.

Then, from the repository root, create the environment and install the local helpers:

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

## Get your API key

The simulations run on the WaveSimPro cloud, so the `wavesimpro` client needs an API key to
authenticate. Once you've [registered](https://studio.rayfos.com/), get one here:

1. Sign in and open **[studio.rayfos.com/api-access](https://studio.rayfos.com/api-access)**.
2. Use a key you already have, or click **Create** to generate a new one, and copy it.

The first time you run a simulation, the `wavesimpro` client detects that it isn't configured yet
and prints instructions on how to register your key (there is more than one supported way). Follow
that prompt with the key you just copied — after that, the examples run without further setup.

## Run an example

```bash
python examples/helmholtz_simulate.py
```

For a guided tour of all the scripts — application showcases, getting-started
building blocks, and validation cases — see the **[Examples Catalogue](EXAMPLES.md)**.

## Layout

```
wavesimpro-starter-kit/
├── pyproject.toml
├── environment.yml   # conda env (conda-forge stack + wavesimpro via pip)
├── requirements.txt  # pip-only install path
├── README.md
├── EXAMPLES.md       # catalogue of every example, grouped and described
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

The waveguide mode solver in [`wsutils/svmodes.py`](wsutils/svmodes.py) (the `svmodes`
finite-difference semi-vectorial eigenmode solver) is vendored from the MIT-licensed
[`modesolver`](https://github.com/thomas-e-murphy/modesolver) package
(© 2025 Thomas E. Murphy), which implements the method of A. B. Fallahkhair, K. S. Li and
T. E. Murphy, *"Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides"*,
J. Lightwave Technol. **26**(11), 1423–1431 (2008),
[doi:10.1109/JLT.2008.923643](https://doi.org/10.1109/JLT.2008.923643). Its full copyright and
permission notice are retained in the `LICENSE` file under *Third-party components*.
