"""
Sub-pixel averaging: a sphere on a coarse grid
==============================================
A plane wave scatters off a dielectric sphere (Mie scattering). The sphere is solved twice
on the same grid: once drawn as a staircase of whole voxels, once with sub-pixel averaging,
where each voxel the surface cuts gets the permittivity averaged over the part of it inside
the sphere. Both scattered fields are compared with the analytical Mie field.

Size: 3 solves, each 4.10 Mvoxel including boundaries, 0.214 GB of memory (service estimate).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from wavesimpro import Setup, Sphere, submit
from wsutils import relative_error, apodise, output_dir

NAME = "mx_subpixel"  # run name on the service and output folder
OUTPUT_DIR = output_dir(NAME)

# Parameters, lengths in micrometer (μm)
wavelength = 1.0
pixel_size = wavelength / 5
n = 120  # voxels per axis
boundary_width = 4.0
sphere_radius = 6.0
sphere_index = 1.2  # refractive index; the background is vacuum
centre = -pixel_size / 2  # sphere centre on every axis; the origin is the domain centre

# A y-polarized plane wave travelling along +z, launched from the first z plane with its
# edges tapered; its phase is zero at the sphere centre
k = 2 * np.pi / wavelength
z_source = (0.5 - n / 2) * pixel_size
values = np.full((1, n, n, 1), 2 * k / (1j * pixel_size) * np.exp(1j * k * (z_source - centre)))
source = apodise((values, [1, 0, 0, 0]), axes="xy", alpha=0.5)  # position: (polarization, ix, iy, iz)


def solve(tag, *shapes):
    """One solve with the source and the given shapes; returns the full field (3, n, n, n)."""
    setup = Setup(wavelength, pixel_size, (n, n, n), boundary_width=boundary_width,
                  name=f"{NAME} {tag}", folder=NAME).add(source, *shapes)
    run = submit(setup).wait()
    print(f"{tag}: {run.iterations} iterations, residual {run.residual_norm:.2e}")
    return run.get_field()


u_background = solve("background")
scattered = {}
for subpixel in (False, True):
    sphere = Sphere(n=sphere_index, center=(centre,) * 3, radius=sphere_radius, subpixel=subpixel)
    tag = "subpixel on" if subpixel else "subpixel off"
    u = solve(tag, sphere) - u_background
    scattered[tag] = u[1, n // 2]  # E_y on the y-z plane through the sphere

# Analytical Mie scattered field, E_y on the same plane and grid
u_ref = np.squeeze(loadmat("examples/wavesim_mat_to_py/matlab_results.mat")["maxwell_mie"])

errors = {tag: relative_error(u, u_ref) for tag, u in scattered.items()}
for tag, re in errors.items():
    print(f"Relative error, {tag}: {re:.2e}")

# Reference field, then the deviation of each solve from it on one colour scale
extent = [-n / 2 * pixel_size, n / 2 * pixel_size] * 2
deviation = {tag: np.abs(u - u_ref) for tag, u in scattered.items()}
vmax = max(d.max() for d in deviation.values())
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
im = axes[0].imshow(np.real(u_ref).T, cmap="RdBu_r", origin="lower", extent=extent)
axes[0].set_title("Mie reference, Re($E_y$) scattered")
fig.colorbar(im, ax=axes[0], fraction=0.046)
for ax, (tag, d) in zip(axes[1:], deviation.items()):
    im = ax.imshow(d.T, cmap="inferno", origin="lower", extent=extent, vmin=0, vmax=vmax)
    ax.set_title(f"|deviation|, {tag}: error {errors[tag]:.2e}")
fig.colorbar(im, ax=axes[1:], fraction=0.046)
for ax in axes:
    ax.add_patch(plt.Circle((centre, centre), sphere_radius, fill=False, ec="k", ls="--", lw=0.8))
    ax.set_xlabel(r"$y~(\mu m)$")
    ax.set_ylabel(r"$z~(\mu m)$")
fig.savefig(f"{OUTPUT_DIR}/{NAME}.png", dpi=150)
plt.show()
