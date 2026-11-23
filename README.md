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
  - `create_source` — `point_source`, `plane_wave`, `gaussian_beam`, `apodise`, `make_directional`
  - `plotting` — `plot_computed`, `plot_computed_and_reference`, `output_dir`
  - `utilities` — `relative_error`, `all_close`, `analytical_solution`, `normalize`, …
  - `power` — `power_flux`, `absorbed_power`
  - `propagation` — `propagate`: carry a field through a homogeneous medium by its angular spectrum
  - `gds` — `write_gds`, `circle`: write polygons to a GDSII layout file
- **`examples/`** — runnable scripts that solve the Helmholtz equation and Maxwell's equations
  on the WaveSimPro cloud.

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

## Quick start

1. **Sign up** at [studio.rayfos.com](https://studio.rayfos.com/).
2. **Create an API key** at [studio.rayfos.com/api-access](https://studio.rayfos.com/api-access)
   and copy it.
3. **Install** the kit (Python 3.11 or later) with conda or pip, as described in
   [Install](#install) below.
4. **Register the key:** run `python -m wavesimpro setup` and paste it. Alternatively, set the
   environment variable `RAYFOS_API_KEY`, or put it in a `.env` file in the folder you run from.
5. **Run an example** from the repository root:
   ```bash
   python examples/hmhz_simulate.py
   ```
   The figure is saved to `output/hmhz_simulate/`, and the run appears in Studio
   ([studio.rayfos.com](https://studio.rayfos.com/)) under the script's name.

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
Nothing runs on your local GPU, so there is no local GPU dependency to install.

## Examples

For a guided tour of all the scripts — getting-started building blocks and validation
cases — see the **[Examples Catalogue](EXAMPLES.md)**. The advanced examples — metalenses,
image-sensor pixels, scattering media, photonic-crystal and integrated-photonics devices, and a
few-mode fiber — are in the companion repository
**[`wavesimpro-examples`](https://github.com/RayFos-Ltd/wavesimpro-examples)**, as paired
scripts and executed notebooks.

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
│   ├── utilities.py
│   ├── power.py
│   ├── propagation.py
│   └── gds.py
└── examples/         # runnable scripts and the layout, mesh and data files they read
```

## Notes

- `wsutils` was extracted from `wavesim.utilities`. Engine-only helpers that were **not** needed
  by these examples (`add_absorbing_boundaries`, `full_matrix`) were dropped, and the few engine
  type-hints were replaced with plain NumPy types. `all_close` accepts either a NumPy array or any
  object exposing a `.gather()` method (e.g. a wavesim engine array), via duck typing.

## License

[LGPL-2.1](LICENSE). You may use `wsutils` and these examples from your own code, open or
proprietary, without obligation; a modified copy of `wsutils` that you redistribute must be
published under the same license. The `wsutils` helpers are adapted from the MIT-licensed
[`wavesim_py`](https://github.com/IvoVellekoop/wavesim_py) project (Ivo Vellekoop, Swapnil Mache —
University of Twente); the `LICENSE` file retains that notice. Running simulations requires a
WaveSimPro account; use of the service is subject to its own terms.
