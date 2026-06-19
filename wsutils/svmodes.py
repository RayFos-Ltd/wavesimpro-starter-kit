"""
Semi-vectorial waveguide mode solver
====================================
A single-function, engine-free extract: the ``svmodes`` finite-difference
semi-vectorial eigenmode solver for a 2D waveguide cross-section. Given a
permittivity profile on the input plane it returns the mode field(s) and the
corresponding effective index/indices, which the examples use to launch a true
guided mode (instead of a Gaussian approximation).

Translated from the MATLAB "Waveguide Mode Solver" by Thomas Murphy:
https://mathworks.com/matlabcentral/fileexchange/12734-waveguide-mode-solver
"""

import numpy as np
from typing import Sequence
from scipy.sparse.linalg import eigs
from scipy.sparse import csr_matrix, eye_array

from .create_source import expand_source_shape, check_source_parameters, polarization_


def svmodes(wavelength: float, guess: complex, nmodes: int, dx: float, dy: float, eps: np.ndarray, field: str):
    """
    Args:
        wavelength: wavelength in micrometer (μm)
        guess: initial guess for effective index (refractive index of the waveguide)
        nmodes: number of modes to compute (usually fundamental, and thus, = 1)
        dx: horizontal grid spacing. Assumed to be scalar
        dy: vertical grid spacing. Assumed to be scalar
        eps: permittivity profile at the input plane
        field: 'TM' mode (EX), 'TE' mode (EY), or 'scalar'
    Returns:
        phi: (nx, ny, nmodes) complex array of mode field(s), each normalized so max|phi| = 1
        neff: effective index/indices of the computed mode(s)
    """
    nx, ny = eps.shape

    # now we pad eps on all sides by one grid point
    eps = np.pad(eps, 1, mode="edge")

    # compute free-space wavevector
    k = 2 * np.pi / wavelength

    dx = dx * np.ones((nx + 2, 1))
    dy = dy * np.ones((1, ny + 2))

    n = np.ones((1, nx * ny), dtype=np.complex64)
    n[:] = (np.ones((nx, 1), dtype=np.complex64) @ (dy[:, 2 : ny + 2] + dy[:, 1 : ny + 1]) / 2).flatten("F")

    s = np.ones((1, nx * ny), dtype=np.complex64)
    s[:] = (np.ones((nx, 1), dtype=np.complex64) @ (dy[:, 0:ny] + dy[:, 1 : ny + 1]) / 2).flatten("F")

    e = np.ones((1, nx * ny), dtype=np.complex64)
    e[:] = ((dx[2 : nx + 2] + dx[1 : nx + 1]) / 2 @ np.ones((1, ny), dtype=np.complex64)).flatten("F")

    w = np.ones((1, nx * ny), dtype=np.complex64)
    w[:] = ((dx[0:nx] + dx[1 : nx + 1]) / 2 @ np.ones((1, ny), dtype=np.complex64)).flatten("F")

    p = np.ones((1, nx * ny), dtype=np.complex64)
    p[:] = (dx[1 : nx + 1] @ np.ones((1, ny), dtype=np.complex64)).flatten("F")

    q = np.ones((1, nx * ny), dtype=np.complex64)
    q[:] = (np.ones((nx, 1), dtype=np.complex64) @ dy[:, 1 : ny + 1]).flatten("F")

    en = np.ones((1, nx * ny), dtype=np.complex64)
    en[:] = eps[1 : nx + 1, 2 : ny + 2].flatten("F")

    es = np.ones((1, nx * ny), dtype=np.complex64)
    es[:] = eps[1 : nx + 1, 0:ny].flatten("F")

    ee = np.ones((1, nx * ny), dtype=np.complex64)
    ee[:] = eps[2 : nx + 2, 1 : ny + 1].flatten("F")

    ew = np.ones((1, nx * ny), dtype=np.complex64)
    ew[:] = eps[0:nx, 1 : ny + 1].flatten("F")

    ep = np.ones((1, nx * ny), dtype=np.complex64)
    ep[:] = eps[1 : nx + 1, 1 : ny + 1].flatten("F")

    if field.lower() == "tm":
        an = 2 / n / (n + s)
        as_ = 2 / s / (n + s)
        ae = (
            8
            * (p * (ep - ew) + 2 * w * ew)
            * ee
            / (
                (p * (ep - ee) + 2 * e * ee) * (p**2 * (ep - ew) + 4 * w**2 * ew)
                + (p * (ep - ew) + 2 * w * ew) * (p**2 * (ep - ee) + 4 * e**2 * ee)
            )
        )
        aw = (
            8
            * (p * (ep - ee) + 2 * e * ee)
            * ew
            / (
                (p * (ep - ee) + 2 * e * ee) * (p**2 * (ep - ew) + 4 * w**2 * ew)
                + (p * (ep - ew) + 2 * w * ew) * (p**2 * (ep - ee) + 4 * e**2 * ee)
            )
        )
        ap = ep * k**2 - an - as_ - (ae * ep / ee) - (aw * ep / ew)
    elif field.lower() == "te":
        an = (
            8
            * (q * (ep - es) + 2 * s * es)
            * en
            / (
                (q * (ep - en) + 2 * n * en) * (q**2 * (ep - es) + 4 * s**2 * es)
                + (q * (ep - es) + 2 * s * es) * (q**2 * (ep - en) + 4 * n**2 * en)
            )
        )
        as_ = (
            8
            * (q * (ep - en) + 2 * n * en)
            * es
            / (
                (q * (ep - en) + 2 * n * en) * (q**2 * (ep - es) + 4 * s**2 * es)
                + (q * (ep - es) + 2 * s * es) * (q**2 * (ep - en) + 4 * n**2 * en)
            )
        )
        ae = 2 / e / (e + w)
        aw = 2 / w / (e + w)
        ap = ep * k**2 - (an * ep / en) - (as_ * ep / es) - ae - aw
    elif field.lower() == "scalar":
        an = 2 / n / (n + s)
        as_ = 2 / s / (n + s)
        ae = 2 / e / (e + w)
        aw = 2 / w / (e + w)
        ap = ep * k**2 - an - as_ - ae - aw
    else:
        raise ValueError(f"field should be 'TM', 'TE', or 'scalar'. Invalid field: '{field}'")

    ii = np.zeros((nx, ny), dtype=int)
    ii[:] = np.reshape(np.arange(0, nx * ny, dtype=int), (nx, ny), order="F")

    iall = np.zeros((1, nx * ny), dtype=int)
    iall[:] = ii.flatten("F")

    in_ = np.zeros((1, nx * (ny - 1)), dtype=int)
    in_[:] = ii[0:nx, 1:ny].flatten("F")

    is_ = np.zeros((1, nx * (ny - 1)), dtype=int)
    is_[:] = ii[0:nx, 0 : (ny - 1)].flatten("F")

    ie = np.zeros((1, (nx - 1) * ny), dtype=int)
    ie[:] = ii[1:nx, 0:ny].flatten("F")

    iw = np.zeros((1, (nx - 1) * ny), dtype=int)
    iw[:] = ii[0 : (nx - 1), 0:ny].flatten("F")

    A = csr_matrix(
        (
            np.concatenate((ap[0, iall], ae[0, iw], aw[0, ie], an[0, is_], as_[0, in_]), axis=1).squeeze(),
            (
                np.concatenate((iall, iw, ie, is_, in_), axis=1).squeeze(),
                np.concatenate((iall, ie, iw, in_, is_), axis=1).squeeze(),
            ),
        )
    )

    shift = (2 * np.pi * guess / wavelength) ** 2

    # [v,d] = eigs(A,speye(size(A)),nmodes,shift,options);
    d, v = eigs(A=A, k=nmodes, M=eye_array(*A.shape, dtype=A.dtype), sigma=shift, tol=1.0e-8)

    neff = wavelength * np.sqrt(np.diag(d)) / (2 * np.pi)
    phi = np.zeros((nx, ny, nmodes), dtype=np.complex64)
    temp = np.zeros((nx, ny), dtype=np.complex64)
    for k in range(0, nmodes):
        temp[:] = np.reshape(-v[:, k] / np.max(np.abs(v[:, k])), (nx, ny), order="F")
        phi[:, :, k] = temp

    return phi, neff


# -----------------------------------------------------------------------------
# Starter-kit convenience wrappers
# -----------------------------------------------------------------------------
# The examples here are 2D, so the waveguide cross-section transverse to the
# propagation direction is a single (1D) axis: the guide width for a top-down
# (x-y) layout, or the guide thickness for a side-view (x-z) layout. The helpers
# below run the 2D ``svmodes`` solver on a thin, transversely-invariant slab to
# obtain that 1D fundamental mode, then package it as a wavesimpro source.


def solve_waveguide_modes(
    index_profile: Sequence[complex],
    pixel_size: float,
    wavelength: float,
    nmodes: int = 1,
    field: str = "scalar",
    guess: complex = None,
):
    """Solve the lowest-order guided modes of a 1D transverse refractive-index profile.

    ``svmodes`` is a 2D cross-section solver, so to obtain *1D* transverse modes
    (the guide width for a top-down x-y layout, or the guide thickness for a
    side-view x-z layout of a 2D simulation) this builds a 1-pixel-wide
    cross-section and gives that second axis a very large grid spacing. The large
    spacing drives the second axis's ``1/dy**2`` finite-difference term to ~0, so
    it contributes nothing to ``beta**2`` and the eigenvalue reduces to the pure
    1D lateral problem (verified to reproduce a direct 1D finite-difference solve
    to ~4 digits for every guided mode). A single column is used so higher-order
    modes are not spuriously triplicated by a multi-pixel invariant axis.

    Modes are returned sorted by descending effective index (mode 0 = fundamental),
    which is what multimode-interference beat-length design needs.

    Args:
        index_profile: 1D refractive-index (n, not n^2) sampled across the
            transverse axis, on the same grid the source will occupy.
        pixel_size: grid spacing in micrometer (μm) along the transverse axis.
        wavelength: wavelength in micrometer (μm).
        nmodes: number of lowest-order modes to return.
        field: 'TE' (E along the transverse axis), 'TM', or 'scalar' (default).
        guess: effective-index guess for the shift-invert eigensolve. Defaults to
            the peak refractive index of ``index_profile`` (the core index).
    Returns:
        phi: (len(index_profile), nmodes) complex mode amplitudes, each column
            normalized to max|phi| = 1, ordered by descending neff.
        neff: length-``nmodes`` complex array of effective indices (descending).
    """
    n_profile = np.asarray(index_profile, dtype=np.complex64).ravel()
    eps2d = (n_profile**2)[:, None].astype(np.complex64)  # single invariant column
    if guess is None:
        guess = float(np.real(n_profile).max())

    # Decouple the invariant axis: spacing >> 1 so its curvature term ~0.
    invariant_spacing = 1000.0
    phi, neff = svmodes(wavelength, guess, int(nmodes), pixel_size, invariant_spacing, eps2d, field)

    neff = np.diag(neff)  # (nmodes,)
    phi_1d = phi[:, 0, :]  # the single column -> (len, nmodes)

    order = np.argsort(-neff.real)  # fundamental (largest neff) first
    neff = neff[order]
    phi_1d = phi_1d[:, order]
    for k in range(phi_1d.shape[1]):
        peak = np.max(np.abs(phi_1d[:, k]))
        if peak > 0:
            phi_1d[:, k] = phi_1d[:, k] / peak
    return phi_1d.astype(np.complex64), neff.astype(np.complex64)


def solve_waveguide_mode(
    index_profile: Sequence[complex],
    pixel_size: float,
    wavelength: float,
    field: str = "scalar",
    guess: complex = None,
):
    """Solve the fundamental guided mode of a 1D transverse refractive-index profile.

    Thin wrapper over ``solve_waveguide_modes`` (nmodes=1) returning the single
    fundamental profile and its effective index. See that function for details.

    Returns:
        phi_1d: 1D complex mode amplitude, same length as index_profile, max|phi|=1.
        neff: complex effective index of the fundamental mode.
    """
    phi, neff = solve_waveguide_modes(index_profile, pixel_size, wavelength, nmodes=1, field=field, guess=guess)
    return phi[:, 0], complex(neff[0])


def mode_source(
    mode_profile: Sequence[complex],
    origin: str,
    position: Sequence[float],
    source_plane: str,
    pixel_size: float,
    amplitude: complex = 1.0,
):
    """Package a precomputed 1D guided-mode profile as a wavesimpro source.

    Mirrors the (source_values, position) convention of ``wsutils.create_source``
    (e.g. ``gaussian_beam``), so the result can be passed straight to
    ``simulate(sources=[mode_source(...)])``. Only single-axis source planes
    ('x', 'y', or 'z') are supported, matching the 1D transverse mode of a 2D
    simulation.

    Args:
        mode_profile: 1D complex mode amplitude (e.g. from ``solve_waveguide_mode``).
        origin: 'center' or 'topleft' -- how ``position`` locates the profile along
            the transverse (source_plane) axis.
        position: 3 floats (x, y, z) for scalar/Helmholtz, or 4 floats
            (polarization axis, x, y, z) for vector/Maxwell, in micrometer (μm).
        source_plane: the single transverse axis the profile spans: 'x', 'y', or 'z'.
        pixel_size: grid spacing in micrometer (μm).
        amplitude: scalar to scale the source by. Default 1.0.
    Returns:
        source_values, position: same convention as the create_source helpers.
    """
    if origin not in ["center", "topleft"]:
        raise ValueError(f"Specify origin from the options ['center', 'topleft']. Invalid origin: '{origin}'")
    source_plane = "".join(sorted(source_plane.lower()))
    if source_plane not in ["x", "y", "z"]:
        raise ValueError(f"mode_source supports a single transverse axis 'x', 'y', or 'z'. Invalid source_plane: '{source_plane}'")

    check_source_parameters(position)
    p = polarization_(position)

    source_values = np.asarray(mode_profile, dtype=np.complex64).ravel() * amplitude
    m = source_values.shape[0]
    source_values = expand_source_shape(source_values, source_plane)
    source_values = source_values[None, ...] if p else source_values  # add polarization dim if present

    position = np.asarray(position)
    position[p:] = position[p:] / pixel_size
    if origin == "center":
        position[p + "xyz".index(source_plane)] -= m // 2
    position = tuple(position.astype(int))

    return source_values.astype(np.complex64), position
