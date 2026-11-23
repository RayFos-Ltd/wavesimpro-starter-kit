"""
GDSII import and mode source: a silicon Y-branch splitter
=========================================================
A Y-branch splits the light of one silicon strip waveguide equally into two. The layout is
read from a GDSII file, ybranch.gds, next to this script: layer 1 holds one polygon, a
0.45 μm input strip that widens to 0.9 μm and divides into two S-bend arms ending in two
0.45 μm output strips 1.5 μm apart. It is extruded to a 220 nm silicon core in SiO2.

The engine solves the input strip's quasi-TE fundamental mode and launches it. A mode port
across both outputs reads the transmitted power; a thin monitor returns the mid-core plane
for the figure and the split.

The voxels are anisotropic: 50 nm along the device and 25 x 22 nm across it, half the
voxels that 25 nm along the device would take.

Size: 34.8 Mvoxel including boundaries, 1.81 GB of memory (service estimate).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from wavesimpro import Gds, Medium, ModePort, Setup, create_mode_source, create_monitor, submit
from wsutils import output_dir

NAME = "mx_gds_ybranch"  # run name on the service and output folder
OUTPUT_DIR = output_dir(NAME)

# Parameters, lengths in micrometer (μm). Axes: x along the device, y across it, z vertical.
wavelength = 1.55
n_si, n_clad = 3.476, 1.444
t_core = 0.22
# Anisotropic voxels (x, y, z): 13 per guided wavelength along x; across the device,
# 18 over a 0.45 μm strip and 10 through the core
pixel_size = [0.05, 0.025, 0.022]
domain = (17.0, 4.5, 2.64)
boundary_widths = [4 * wavelength, 1 * wavelength, 0.5 * wavelength]
nx, ny, nz = (int(round(s / p)) for s, p in zip(domain, pixel_size))

# The layout's coordinates are centred on the domain: the junction is at x = -1.5 and the
# arms reach their 1.5 μm spacing at x = 4.5. Its strips run on through the absorbing
# boundaries, so the light meets no end facet.
x_out, y_out = 4.5, 0.75
i_of = lambda x: int(round((x + domain[0] / 2) / pixel_size[0]))  # voxel index along x

# The service places the centre of the layout's bounding box at `center`; that box is
# symmetric about the layout's origin. A layer's z offset is the height of its bottom face.
silicon = Medium("Si", n_si)
layout = Gds(n=silicon, name="ybranch", path=Path(__file__).with_name("ybranch.gds"),
             center=(0.0, 0.0, 0.0), layers={1: (silicon, t_core, -t_core / 2)},
             subpixel=True)  # voxels the sidewalls cut get the averaged permittivity

# Quasi-TE fundamental mode of the input strip, driven over 3 μm ending at x = -5
source = create_mode_source(name="TE0", axis="x", plane_index=i_of(-5.0), direction=+1,
                            polarization="TE", order=0, drive_length_um=3.0,
                            engine_solver="ps_operator")

# The output pair's two lowest quasi-TE modes are its symmetric (TE0) and antisymmetric (TE1)
# supermodes; their forward powers add up to the transmission.
port = ModePort("outputs", axis="x", plane_index=i_of(x_out + 2.5), modes=["TE0", "TE1"],
                engine_solver="fd")

# Three voxels around mid-core height; voxel i sits at (i + 0.5 - nz/2) * dz
z_mid = (nz // 2 + 0.5 - nz / 2) * pixel_size[2]
monitor = create_monitor(name="midplane", center_x=0.0, center_y=0.0, center_z=z_mid,
                         size_x=domain[0], size_y=domain[1], size_z=3 * pixel_size[2])

setup = Setup(wavelength, pixel_size, (nx, ny, nz), boundary_widths=boundary_widths,
              background=Medium("SiO2", n_clad), name=NAME)
setup.add(layout, source, port, monitor)
run = submit(setup).wait()
print(f"{run.iterations} iterations, residual {run.residual_norm:.2e}")

intensity = np.abs(run.get_monitors()[0][1, :, :, 1]) ** 2  # |E_y|^2 on the mid-core plane
x = (np.arange(nx) + 0.5) * pixel_size[0] - domain[0] / 2
y = (np.arange(ny) + 0.5) * pixel_size[1] - domain[1] / 2

# Split: power in each output strip just after the bends
i = i_of(x_out + 0.5)
upper = intensity[i, np.abs(y - y_out) < 0.4].sum()
lower = intensity[i, np.abs(y + y_out) < 0.4].sum()
split = upper / (upper + lower)
forward = {m["Label"]: m["ForwardPower"] for m in run.port_results()["Ports"][0]["Modes"]}
transmission = forward["TE0"] + forward["TE1"]
print(f"Split {split:.1%} / {1 - split:.1%}; transmission {transmission:.1%} "
      f"(TE0 {forward['TE0']:.1%}, TE1 {forward['TE1']:.1%})")

plt.figure(figsize=(11, 4))
plt.imshow(intensity.T / intensity.max(), cmap="inferno", origin="lower", norm=LogNorm(1e-3, 1),
           extent=[x[0], x[-1], y[0], y[-1]])
plt.colorbar(fraction=0.02, pad=0.02, label=r"$|E_y|^2$ (normalized)")
plt.xlabel(r"$x~(\mu m)$")
plt.ylabel(r"$y~(\mu m)$")
plt.title(f"Y-branch: split {split:.1%} / {1 - split:.1%}, transmission {transmission:.1%}")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/{NAME}.png", dpi=150)
plt.show()
