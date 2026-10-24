# WaveSimPro Starter-Kit — Examples Catalogue
<div align="center">

[![WaveSimPro examples demo](https://img.youtube.com/vi/LcBW98BSRhQ/maxresdefault.jpg)](https://youtu.be/LcBW98BSRhQ)

▶️ Click to watch the demo on YouTube

</div>
A guided tour of every script in [`examples/`](examples/). All examples call
`from wavesimpro import simulate` and run on the WaveSimPro cloud (see the
[README](README.md) for installation and API-key setup).

Every example is a **single-frequency, continuous-wave** solve — the result is a
steady-state complex field (amplitude **and** phase), not a time-domain animation.
Scripts are either **scalar Helmholtz** (one field component) or **vector Maxwell** (the
full E-field), and run in 2-D or 3-D as noted. Figures go to a git-ignored `output/` folder.

The examples fall into two groups, with a pointer to the advanced ones:

1. [Getting started & building blocks](#1-getting-started--building-blocks) — short
   scripts that show how to call `simulate` and how to build sources, media and monitors.
2. [Accuracy & validation](#2-accuracy--validation) — solves checked against analytic
   theory and saved reference fields.
3. [Advanced examples](#3-advanced-examples) — realistic devices, in a companion repository.

---

## 1. Getting started & building blocks

Short, self-contained scripts that introduce the `simulate` call and the helper
utilities in `wsutils` (sources and media). Good first reads.

| Example | Solves | What it shows |
|---------|--------|---------------|
| [`helmholtz_simulate.py`](examples/helmholtz_simulate.py) | Helmholtz | A tour of **every argument** of the `simulate` wrapper, solving for a point source in a random medium. |
| [`maxwell_simulate.py`](examples/maxwell_simulate.py) | Maxwell | The same tour of `simulate` arguments, solving the vector Maxwell field. |
| [`run_example.py`](examples/run_example.py) | Helmholtz | A minimal end-to-end script: a point source in a random refractive-index map. |
| [`helmholtz_3d_create_source.py`](examples/helmholtz_3d_create_source.py) | Helmholtz 3-D | The three source primitives — `point_source`, `plane_wave`, `gaussian_beam` — launched across an interface between two media. |
| [`helmholtz_1d_create_medium.py`](examples/helmholtz_1d_create_medium.py) | Helmholtz 1-D | Building a medium from material blocks with `cuboids_permittivity` and propagating through it. |
| [`maxwell_2d_create_medium.py`](examples/maxwell_2d_create_medium.py) | Maxwell 2-D | Building a 2-D medium of material blocks and solving Maxwell's equations. |
| [`helmholtz_3d_advanced.py`](examples/helmholtz_3d_advanced.py) | Helmholtz 3-D | Advanced usage: **multiple sources** in a 3-D block medium, plus a progress **callback**. |
| [`helmholtz_3d_import_lens.py`](examples/helmholtz_3d_import_lens.py) | Helmholtz 3-D | Loading a refractive-index map from a `.mat` file and propagating a Gaussian beam through the lens. |
| [`hello_monitors.py`](examples/hello_monitors.py) | Maxwell 3-D | **Monitors**: ask `simulate` for two regions of the field (a plane and a small cube) instead of the whole volume, and read them back from the list it returns. |
| [`maxwell_waveguide_basic.py`](examples/maxwell_waveguide_basic.py) | Maxwell 2-D | A first **waveguide**: a Gaussian beam launched into a dielectric strip stays guided along it. The companion repository takes this on to couplers and GDSII devices. |

---

## 2. Accuracy & validation

Solves compared against analytic theory or saved reference fields (from the original
MATLAB `wavesim` repository) using `relative_error`. The scripts in
[`examples/wavesim_mat_to_py/`](examples/wavesim_mat_to_py/) load their reference
`.mat` files by relative path, so **run them from the repository root**
(e.g. `python examples/wavesim_mat_to_py/maxwell_mie.py`).

| Example | Solves | Compared against |
|---------|--------|------------------|
| [`wavesim_mat_to_py/maxwell_mie.py`](examples/wavesim_mat_to_py/maxwell_mie.py) | Maxwell 3-D | **Mie scattering** from a dielectric sphere vs analytic Mie theory — the precision benchmark. |
| [`helmholtz_1d_analytical.py`](examples/helmholtz_1d_analytical.py) | Helmholtz 1-D | 1-D free-space propagation vs the analytic solution. |
| [`wavesim_mat_to_py/helmholtz_1d.py`](examples/wavesim_mat_to_py/helmholtz_1d.py) | Helmholtz 1-D | 1-D propagation through a glass plate vs a MATLAB reference. |
| [`wavesim_mat_to_py/helmholtz_2d_low_contrast.py`](examples/wavesim_mat_to_py/helmholtz_2d_low_contrast.py) | Helmholtz 2-D | A low-contrast medium (fat/water, tissue-like) vs a MATLAB reference. |
| [`wavesim_mat_to_py/helmholtz_2d_high_contrast.py`](examples/wavesim_mat_to_py/helmholtz_2d_high_contrast.py) | Helmholtz 2-D | A high-contrast medium (iron) vs a MATLAB reference. |
| [`wavesim_mat_to_py/helmholtz_3d_disordered.py`](examples/wavesim_mat_to_py/helmholtz_3d_disordered.py) | Helmholtz 3-D | A 3-D disordered medium vs a MATLAB reference. |
| [`wavesim_mat_to_py/maxwell_2d.py`](examples/wavesim_mat_to_py/maxwell_2d.py) | Maxwell 2-D | 2-D Maxwell propagation vs a MATLAB reference. |

---

## 3. Advanced examples

Realistic, wavelength-scale devices — flat and nanopillar metalenses, a CMOS microlens,
photonic crystals and a bandgap waveguide bend, a computer-generated hologram, Bessel and
vortex beams, random-microsphere scattering, image-sensor light trapping, and integrated
photonics (grating coupler, MMI, directional coupler, GDSII-imported devices) — live in the
companion repository **`wavesimpro-examples`**, to be published.
They are notebooks and use the same `wsutils` helpers as this kit.
