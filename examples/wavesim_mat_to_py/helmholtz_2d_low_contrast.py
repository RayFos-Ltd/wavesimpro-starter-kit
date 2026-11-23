""" 
Helmholtz 2D low contrast test
===============================
Test for propagation in 2D structure with low refractive index contrast 
(made of fat and water to mimic biological tissue). 
Compare with the reference solution computed by the MATLAB WaveSim implementation.

Size: 0.250 Mvoxel including boundaries, 0.008 GB of memory (service estimate).
"""

import numpy as np
from time import time
from scipy.io import loadmat
from PIL.Image import fromarray, open, Resampling

from wavesimpro import simulate, configure
from wsutils import relative_error, plot_computed_and_reference, output_dir

NAME = "helmholtz_2d_low_contrast"  # run name on the service and output folder
OUTPUT_DIR = output_dir(NAME)
configure(description=NAME)

# Parameters
n_water = 1.33
n_fat = 1.46
wavelength = 0.532  # Wavelength in micrometer (μm)
pixel_size = wavelength / (3 * abs(n_fat))  # Pixel size in micrometer (μm)

# Load image and create refractive index map
oversampling = 0.25
im = np.asarray(open("examples/wavesim_mat_to_py/logo_structure_vector.png")) / 255  # Load image and normalize to [0, 1]
n_im = (np.where(im[:, :, 2] > 0.25, 1, 0) * (n_fat - n_water)) + n_water  # Assign refractive index to pixels
n_roi = int(oversampling * n_im.shape[0])  # Size of ROI in pixels
permittivity = np.asarray(fromarray(n_im).resize((n_roi, n_roi), Resampling.BILINEAR))[..., None] ** 2  # permittivity of the domain

# Create a custom source using the array from the imported png
source_values = np.asarray(fromarray(im[:, :, 1]).resize((n_roi, n_roi), Resampling.BILINEAR))[..., None]
source_position = [0, 0, 0]  # source position in (x, y, z) in pixels

# Run the wavesim iteration and get the computed field
start = time()
u, iterations, residual_norm = simulate(
    permittivity=permittivity, 
    sources=[ (source_values, source_position) ], 
    wavelength=wavelength, 
    pixel_size=pixel_size, 
    boundary_width=5,  # Boundary width in micrometer (μm)
    periodic=(False, False, True)
)
sim_time = time() - start
print(f"Time {sim_time:2.2f} s; Iterations {iterations}; Time per iteration {sim_time / iterations:.4f} s")
print(f"(Residual norm {residual_norm:.2e})")

# load the MATLAB reference results
u_ref = np.squeeze(loadmat("examples/wavesim_mat_to_py/matlab_results.mat")["u2d_lc"])

# Compute relative error with respect to the reference solution
re = relative_error(u, u_ref)
print(f"Relative error with reference: {re:.2e}")

# Plot the results
plot_computed_and_reference(u, u_ref, pixel_size, re, save=True, filename=f"{OUTPUT_DIR}/{NAME}")

threshold = 1.0e-3
assert re < threshold, f"Relative error {re} higher than {threshold}"
