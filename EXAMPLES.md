# WaveSimPro Starter-Kit — Examples Catalogue
<div align="center">

[![WaveSimPro examples demo](https://img.youtube.com/vi/LcBW98BSRhQ/maxresdefault.jpg)](https://youtu.be/LcBW98BSRhQ)

▶️ [Watch the demo on YouTube](https://youtu.be/LcBW98BSRhQ)

</div>
A guided tour of every script in [`examples/`](examples/). Most examples call
`simulate` with NumPy arrays; a few describe the device with `Setup`/`submit` and
primitives. All run on the WaveSimPro cloud (see the
[README](README.md) for installation and API-key setup).

Every example is a **single-frequency, continuous-wave** solve — the result is a
steady-state complex field (amplitude **and** phase), not a time-domain animation.
Scripts are either **scalar Helmholtz** (one field component) or **vector Maxwell** (the
full E-field), and run in 2-D or 3-D as noted. Each example saves its figure(s) to
`output/<example name>/` at the repository root (git-ignored) and names its run on the
service after the script.

The **Size** column gives the grid in Mvoxel, including the absorbing boundaries, and the
memory the service estimates for the solve.

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

| Example | Solves | What it shows | Size (Mvoxel / GB) |
|---------|--------|---------------|--------------------|
| [`hmhz_simulate.py`](examples/hmhz_simulate.py) | Helmholtz | A tour of **every argument** of the `simulate` wrapper, solving for a point source in a random medium. | 2.765 / 0.062 |
| [`mx_simulate.py`](examples/mx_simulate.py) | Maxwell | The same tour of `simulate` arguments, solving the vector Maxwell field. | 2.765 / 0.144 |
| [`hmhz_3d_create_source.py`](examples/hmhz_3d_create_source.py) | Helmholtz 3-D | The three source primitives — `point_source`, `plane_wave`, `gaussian_beam` — launched across an interface between two media. | 10.75 / 0.240 |
| [`hmhz_1d_create_medium.py`](examples/hmhz_1d_create_medium.py) | Helmholtz 1-D | Building a medium from material blocks with `cuboids_permittivity` and propagating through it. | 0.0004 / 0.00001 |
| [`mx_2d_create_medium.py`](examples/mx_2d_create_medium.py) | Maxwell 2-D | Building a 2-D medium of material blocks and solving Maxwell's equations. | 0.085 / 0.004 |
| [`hmhz_3d_advanced.py`](examples/hmhz_3d_advanced.py) | Helmholtz 3-D | Advanced usage: **multiple sources** in a 3-D block medium, plus a progress **callback**. | 0.704 / 0.016 |
| [`hmhz_3d_import_lens.py`](examples/hmhz_3d_import_lens.py) | Helmholtz 3-D | Loading a refractive-index map from a `.mat` file and propagating a Gaussian beam through the lens. | 20.80 / 0.465 |
| [`mx_monitors.py`](examples/mx_monitors.py) | Maxwell 3-D | **Monitors**: ask `simulate` for two regions of the field (a plane and a small cube) instead of the whole volume, and read them back from the list it returns. | 0.512 / 0.027 |
| [`mx_subpixel.py`](examples/mx_subpixel.py) | Maxwell 3-D | **Sub-pixel averaging**: a sphere drawn as a `Sphere` primitive and solved with `Setup`/`submit`, with averaging off and on, each compared with the analytical Mie field. | 3 solves, each 4.10 / 0.214 |
| [`mx_gds_ybranch.py`](examples/mx_gds_ybranch.py) | Maxwell 3-D | **GDSII layout**: a silicon Y-branch read from `ybranch.gds` as one `Gds` layer, launched with the guide's own mode and read with a two-mode port: the split between the arms and the total transmission. Solved on **anisotropic voxels**, coarser along the device than across it. | 34.8 / 1.81 |
| [`hmhz_stl_lens.py`](examples/hmhz_stl_lens.py) | Helmholtz 3-D | **STL mesh**: a diffractive lens read from `diffractive_lens.stl`, focusing into a clear liquid and then into a turbid one given as a voxel array. | 2 solves, each 33.7 / 0.76 |
| [`mx_waveguide_basic.py`](examples/mx_waveguide_basic.py) | Maxwell 2-D | A first **waveguide**: a Gaussian beam launched into a dielectric strip stays guided along it. The companion repository takes this on to couplers and splitters. | 0.243 / 0.013 |

---

## 2. Accuracy & validation

Solves compared against analytic theory or saved reference fields (from the original
MATLAB `wavesim` repository) using `relative_error`. The scripts in
[`examples/wavesim_mat_to_py/`](examples/wavesim_mat_to_py/) load their reference
`.mat` files by relative path, so **run them from the repository root**
(e.g. `python examples/wavesim_mat_to_py/maxwell_mie.py`).

| Example | Solves | Compared against | Size (Mvoxel / GB) |
|---------|--------|------------------|--------------------|
| [`wavesim_mat_to_py/maxwell_mie.py`](examples/wavesim_mat_to_py/maxwell_mie.py) | Maxwell 3-D | **Mie scattering** from a dielectric sphere vs analytic Mie theory — the precision benchmark. | 2 solves, each 4.096 / 0.214 |
| [`hmhz_1d_analytical.py`](examples/hmhz_1d_analytical.py) | Helmholtz 1-D | 1-D free-space propagation vs the analytic solution. | 0.0028 / 0.0001 |
| [`wavesim_mat_to_py/helmholtz_1d.py`](examples/wavesim_mat_to_py/helmholtz_1d.py) | Helmholtz 1-D | 1-D propagation through a glass plate vs a MATLAB reference. | 0.0003 / 0.00001 |
| [`wavesim_mat_to_py/helmholtz_2d_low_contrast.py`](examples/wavesim_mat_to_py/helmholtz_2d_low_contrast.py) | Helmholtz 2-D | A low-contrast medium (fat/water, tissue-like) vs a MATLAB reference. | 0.250 / 0.008 |
| [`wavesim_mat_to_py/helmholtz_2d_high_contrast.py`](examples/wavesim_mat_to_py/helmholtz_2d_high_contrast.py) | Helmholtz 2-D | A high-contrast medium (iron) vs a MATLAB reference. | 0.420 / 0.009 |
| [`wavesim_mat_to_py/helmholtz_3d_disordered.py`](examples/wavesim_mat_to_py/helmholtz_3d_disordered.py) | Helmholtz 3-D | A 3-D disordered medium vs a MATLAB reference. | 1.032 / 0.023 |
| [`wavesim_mat_to_py/maxwell_2d.py`](examples/wavesim_mat_to_py/maxwell_2d.py) | Maxwell 2-D | 2-D Maxwell propagation vs a MATLAB reference. | 0.071 / 0.004 |

---

## 3. Advanced examples

The advanced examples — metalenses, image-sensor pixels, scattering media, photonic-crystal
and integrated-photonics devices, and a few-mode fiber — are in the companion repository
**[`wavesimpro-examples`](https://github.com/RayFos-Ltd/wavesimpro-examples)**, as paired
scripts and executed notebooks.
