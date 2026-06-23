# WaveSimPro Starter-Kit — Examples Catalogue

A guided tour of every script in [`examples/`](examples/). All examples call
`from wavesimpro import simulate` and run on the WaveSimPro cloud (see the
[README](README.md) for installation and API-key setup).

Every example is a **single-frequency, continuous-wave** solve — the result is a
steady-state complex field (amplitude **and** phase), not a time-domain animation.
Scripts are either **scalar Helmholtz** (one field component) or **vector Maxwell**
(the full E-field), and run in 2-D or 3-D as noted. The application examples write
their figures and volumes to a git-ignored `output/` folder.

The examples fall into three groups:

1. [Getting started & building blocks](#1-getting-started--building-blocks) — short
   scripts that show how to call `simulate` and how to build sources and media.
2. [Accuracy & validation](#2-accuracy--validation) — solves checked against analytic
   theory and saved reference fields.
3. [Application showcases](#3-application-showcases) — realistic, wavelength-scale
   structures where the wave nature of light is essential.

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

## 3. Application showcases

These solve structures whose features are comparable to the wavelength, so geometric
ray tracing does not apply and the result depends on diffraction, interference and
near-field coupling. Each returns the steady-state field directly.

| Example | Solves | What it demonstrates |
|---------|--------|----------------------|
| [`maxwell_metalens.py`](examples/maxwell_metalens.py) | Maxwell 3-D | A flat, wavelength-thin lens (a blazed surface-relief kinoform) focuses a plane wave to a diffraction-limited Airy spot. Shows the focal cone and reports spot size and NA. |
| [`maxwell_metalens_nanopillar.py`](examples/maxwell_metalens_nanopillar.py) | Maxwell 3-D | The same flat lens built as a true sub-wavelength **nanopillar metasurface**, with the pillar-diameter→phase library calibrated from periodic unit-cell solves — the way real metasurfaces are designed. |
| [`maxwell_metalens_2d.py`](examples/maxwell_metalens_2d.py) | Maxwell 2-D | A fast 2-D cross-section of the flat lens — the quickest way to see a metalens focus (runs in seconds). |
| [`maxwell_microlens.py`](examples/maxwell_microlens.py) | Maxwell 2-D | A CMOS/CCD sensor microlens only a few wavelengths across: diffraction-limited focus, the pixel's angular response over a chief-ray-angle sweep, and optical crosstalk into neighbouring pixels. |
| [`helmholtz_bessel_vortex.py`](examples/helmholtz_bessel_vortex.py) | Helmholtz | A Bessel beam propagated over ~100 wavelengths: a non-diffracting core (next to a Gaussian that spreads), self-healing past an opaque obstacle, and an optical vortex (doughnut intensity + helical phase). |
| [`helmholtz_random_microspheres.py`](examples/helmholtz_random_microspheres.py) | Helmholtz | A plane wave through a slab of randomly packed microspheres develops fully three-dimensional **volumetric speckle** from multiple coherent scattering. |
| [`helmholtz_random_microspheres_opaque.py`](examples/helmholtz_random_microspheres_opaque.py) | Helmholtz | The strong, **opaque/diffusive** regime of the above (high-index, thick, densely packed): far-field speckle contrast and grain size, plus a convergence plot. A large cloud-scale run. |
| [`maxwell_photonic_crystal.py`](examples/maxwell_photonic_crystal.py) | Maxwell 2-D | A finite photonic crystal (square lattice of dielectric rods), wavelength-swept to reveal the **photonic bandgap** — a band of colours the crystal cannot transmit. Reports true transmittance. |
| [`maxwell_pc_waveguide_bend.py`](examples/maxwell_pc_waveguide_bend.py) | Maxwell 2-D | A line-defect waveguide carved into the same crystal guides light around a sharp **90° bend** with near-total transmission — confinement by the bandgap, which a conventional waveguide cannot match. |
| [`helmholtz_hologram_cgh.py`](examples/helmholtz_hologram_cgh.py) | Helmholtz | A flat diffractive phase plate (kinoform), designed by Gerchberg–Saxton, reconstructs a recognizable image in mid-air purely by diffraction. Compares the rigorous full-wave reconstruction against the thin-screen Fourier-optics prediction. |
| [`maxwell_grating_coupler.py`](examples/maxwell_grating_coupler.py) | Maxwell 3-D | The canonical fiber-to-chip **grating coupler** as a full 3-D structure: a finite-width, partially-etched silicon waveguide grating (220 nm Si on buried oxide, air top-cladding) diffracts the guided mode up out of the chip into a free-space beam for an optical fiber. Launched with the true (y, z) waveguide mode from the svmodes solver. |
| [`maxwell_mmi_splitter.py`](examples/maxwell_mmi_splitter.py) | Maxwell 3-D | A 1×2 **multimode-interference (MMI) splitter** as a full 3-D SOI device: a wide multimode section self-images the 220 nm-core input mode into a 50/50 split. The two-image length is designed by modal expansion of the real (y, z) cross-section modes (svmodes) and validated by the full-wave run. |
| [`maxwell_directional_coupler.py`](examples/maxwell_directional_coupler.py) | Maxwell 3-D | A **directional coupler** as a full 3-D SOI device: power sloshes between two side-by-side 450×220 nm silicon waveguides via evanescent supermode beating. The coupling length `L_c` from the svmodes supermodes sets the device length; the full-wave run shows full power crossover at `L_c` and return at `2 L_c`. |
