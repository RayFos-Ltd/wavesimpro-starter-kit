"""
STL geometry: a diffractive lens focusing into a turbid liquid
==============================================================
A flat diffractive (kinoform) lens is read from ``diffractive_lens.stl``: a glass disc of
14 μm diameter with a sawtooth relief, one wavelength of optical path deep against air, on
the side facing the incoming light; its flat side touches a liquid of index 1.33. The zone
edges sit where the optical path from the relief, through the lens and the liquid, to a point
f = 10 μm past the flat face grows by a whole wavelength, so rays from all zones meet there
in green light. That layout treats the relief as a thin phase plate; the solved focus forms
about 0.5 μm past f, inside its 2 μm depth of focus. A plane wave from below is focused
twice with the same lens: into the clear liquid, then into a turbid one with the same mean
index, so the focus stays in place; its random index fluctuations dim the focus and spread
part of the light into speckle.
Scalar (Helmholtz) solves.

Size: 2 solves, each 33.7 Mvoxel including boundaries, 0.76 GB of memory (service estimate).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from wavesimpro import Monitor, Setup, Stl, VoxelArray, submit
from wsutils.create_source import apodise, plane_wave
from wsutils import output_dir

NAME = "hmhz_stl_lens"  # run name on the service and output folder
OUTPUT_DIR = output_dir(NAME)

# Parameters, lengths in micrometer (μm); the origin is the domain centre
wavelength = 0.532
pixel_size = wavelength / 8  # λ/5.3 in the glass
n_xy, n_z = 271, 236  # voxels; odd n_xy puts a voxel centre on the lens axis
size_xy, size_z = n_xy * pixel_size, n_z * pixel_size
boundary_width = 4 * wavelength

# Lens design, as drawn in the STL (mesh units are μm, axis at x = y = 0): the relief at the
# bottom, then a flat base whose top face touches the liquid
lens_index = 1.5
liquid_index = 1.33
lens_radius = 7.0
focal_length = 10.0  # in the liquid, from the flat face
base = 0.5
relief = wavelength / (lens_index - 1)  # a full 2π of phase against air
k_flat = 39  # the flat face lies on this voxel face, leaving 1 μm of air below the lens
z_flat = -size_z / 2 + k_flat * pixel_size
z_base = z_flat - base - relief  # bottom of the lens

# The mesh is placed by the centre of its bounding box
lens = Stl(n=lens_index, path="examples/diffractive_lens.stl",
           center=(0.0, 0.0, z_base + (base + relief) / 2),
           subpixel=True)  # cut voxels get the volume-averaged permittivity

# Liquid from the flat face to the far end of the domain. Its turbid version adds random
# index noise smoothed by a 0.25 μm Gaussian; 0.017 rms dims the focus to about 0.63 of its
# clear peak intensity
n_liquid = n_z - k_flat
rng = np.random.default_rng(seed=1)
noise = rng.standard_normal((n_xy, n_xy, n_liquid))
k2 = sum(np.fft.fftfreq(n, pixel_size).reshape(shape) ** 2
         for n, shape in ((n_xy, (-1, 1, 1)), (n_xy, (1, -1, 1)), (n_liquid, (1, 1, -1))))
noise = np.fft.ifftn(np.fft.fftn(noise) * np.exp(-2 * (np.pi * 0.25) ** 2 * k2)).real
noise /= noise.std()


def liquid(rms):
    """The liquid as a refractive-index array, mean 1.33 with the given rms fluctuation."""
    index = (liquid_index + rms * noise).astype(np.float32)
    return VoxelArray(array=index, data_type="RefractiveIndex",  # placed by its centre
                      center=(0.0, 0.0, z_flat + n_liquid * pixel_size / 2), subpixel=False)


# A plane wave travelling along +z, launched 0.5 μm above the bottom, edges tapered
source = apodise(plane_wave(shape=(size_xy, size_xy), origin="topleft", position=(0, 0, 0.5),
                            source_plane="xy", pixel_size=pixel_size), axes="xy", alpha=0.2)

# The x-z plane through the lens axis, one voxel thick
xz_plane = Monitor("xz_plane", center=(0.0, 0.0, 0.0), size=(size_xy, 0.0, size_z))


def solve(tag, medium):
    """One solve with the lens, the given liquid and the source; returns |u|^2 on the x-z plane."""
    setup = Setup(wavelength, pixel_size, (n_xy, n_xy, n_z), boundary_width=boundary_width,
                  name=f"{NAME} {tag}", folder=NAME).add(lens, medium, source, xz_plane)
    run = submit(setup).wait()
    print(f"{tag}: {run.iterations} iterations, residual {run.residual_norm:.2e}")
    return np.abs(np.squeeze(run.get_monitors()[0])) ** 2


intensity = {"clear liquid": solve("clear liquid", liquid(0.0)),
             "turbid liquid": solve("turbid liquid", liquid(0.017))}

# Focus of each solve: brightest voxel on the axis inside the liquid
x = (np.arange(n_xy) + 0.5 - n_xy / 2) * pixel_size
z = (np.arange(n_z) + 0.5 - n_z / 2) * pixel_size
behind = z > z_flat
sin_glass = lambda theta: liquid_index * np.sin(theta) / lens_index  # Snell at the flat face
theta = brentq(lambda t: (base + relief) * np.tan(np.arcsin(sin_glass(t))) + focal_length * np.tan(t)
               - lens_radius, 0.0, 1.0)  # angle in the liquid of the ray from the lens rim
na = liquid_index * np.sin(theta)
print(f"Design focus {focal_length:.2f} um past the flat face, "
      f"diffraction limit 0.51 lambda/NA = {0.51 * wavelength / na:.3f} um (NA {na:.2f})")
for tag, u in intensity.items():
    iz = np.flatnonzero(behind)[np.argmax(u[n_xy // 2, behind])]
    profile = u[:, iz] / u[n_xy // 2, iz]
    edges = np.flatnonzero(profile >= 0.5)
    left, right = edges[0], edges[-1]  # linear interpolation at the two half-maximum crossings
    fwhm = (x[right] + (profile[right] - 0.5) / (profile[right] - profile[right + 1]) * pixel_size
            - x[left] + (profile[left] - 0.5) / (profile[left] - profile[left - 1]) * pixel_size)
    print(f"{tag}: focus {z[iz] - z_flat:.2f} um past the flat face, FWHM {fwhm:.3f} um")
peaks = {tag: u[:, behind].max() for tag, u in intensity.items()}
print(f"Peak intensity, turbid / clear: {peaks['turbid liquid'] / peaks['clear liquid']:.2f}")

# |u|^2 on the x-z plane relative to the clear focus, both solves on one log scale
extent = [x[0] - pixel_size / 2, x[-1] + pixel_size / 2, z[0] - pixel_size / 2, z[-1] + pixel_size / 2]
fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True, constrained_layout=True)
for ax, (tag, u) in zip(axes, intensity.items()):
    im = ax.imshow(np.log10(u / peaks["clear liquid"]).T, cmap="inferno", origin="lower", extent=extent,
                   vmin=-3, vmax=0)
    ax.axhline(z_flat + focal_length, color="w", ls=":", lw=0.8)  # design focal plane
    ax.add_patch(plt.Rectangle((-lens_radius, z_base), 2 * lens_radius, base + relief, fill=False, ec="c", lw=0.8))
    ax.add_patch(plt.Rectangle((x[0], z_flat), size_xy, size_z / 2 - z_flat, fill=False, ec="c", ls="--", lw=0.8))
    ax.set_title(f"{tag}: peak {peaks[tag] / peaks['clear liquid']:.2f}")
    ax.set_xlabel(r"$x~(\mu m)$")
axes[0].set_ylabel(r"$z~(\mu m)$")
fig.colorbar(im, ax=axes, label=r"$\log_{10}(|u|^2 / $clear focus$)$", fraction=0.046)
fig.savefig(f"{OUTPUT_DIR}/{NAME}.png", dpi=150)
plt.show()
