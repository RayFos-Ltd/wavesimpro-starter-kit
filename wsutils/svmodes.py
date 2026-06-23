"""
Semi-vectorial waveguide mode solver
====================================
An engine-free extract of the ``svmodes`` finite-difference semi-vectorial
eigenmode solver for a 2D waveguide cross-section. Given a permittivity profile
on the input plane it returns the effective index/indices and the mode field(s),
which the examples use to launch a true guided mode (instead of a Gaussian
approximation) and to design modal beat lengths (MMI, directional coupler).

Vendored (the solver function only) from the ``modesolver`` package by
Thomas E. Murphy (tem@umd.edu) — finite-difference method of
    A. B. Fallahkhair, K. S. Li and T. E. Murphy,
    "Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides",
    J. Lightwave Technol. 26(11), 1423-1431 (2008). doi:10.1109/JLT.2008.923643
    https://github.com/thomas-e-murphy/modesolver

Copyright (c) 2025 Thomas E. Murphy. Used and modified under the MIT License; the
full copyright and permission notice are retained in the repository ``LICENSE``
file (see "Third-party components").

Trimmed for this starter kit to a numpy + scipy footprint: the upstream
shift-invert solver auto-selection (PyPardiso / MUMPS) is dropped in favour of
SciPy's built-in shift-invert (``eigs(..., sigma=...)``), which factorizes
``A - sigma*I`` with SuperLU. No extra dependencies are introduced.
"""

import warnings
import numpy as np
from typing import Sequence
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs

from .create_source import expand_source_shape, check_source_parameters, polarization_


def svmodes(wavelength, guess, nmodes, dx, dy, eps, boundary, field):
    """
    Calculate the modes of a dielectric waveguide using the semivectorial
    finite difference method.

    USAGE:
        neff, phi = svmodes(wavelength, guess, nmodes, dx, dy, eps, boundary, field)

    INPUT:
        wavelength : optical wavelength (same length unit as dx, dy)
        guess      : scalar effective-index shift; returns the eigenpairs whose
                     effective index is closest to this guess.
        nmodes     : number of modes to calculate.
        dx         : horizontal grid spacing. Scalar (uniform) or a vector of
                     length nx (number of columns of eps) for a non-uniform grid.
        dy         : vertical grid spacing. Scalar (uniform) or a vector of
                     length ny (number of rows of eps) for a non-uniform grid.
        eps        : index mesh (n^2(x, y)); a 2-D array with shape (ny, nx).
        boundary   : 4-letter string of edge conditions, N, S, E, W:
                       boundary[0] = North, [1] = South, [2] = East, [3] = West.
                       'A' antisymmetric, 'S' symmetric, '0' Dirichlet (zero).
        field      : 'EX' (semivectorial Ex), 'EY' (semivectorial Ey), or
                     'scalar' (scalar Helmholtz approximation).

    OUTPUT:
        neff       : modal effective indices, shape (nmodes,), sorted descending.
        phi        : requested field component per mode, shape (ny, nx, nmodes),
                     each normalized so |phi| peaks at 1. Fields sit at the
                     *center* of each finite-difference cell.
                     If nmodes == 1, phi is 2-D (ny, nx) and neff is a scalar.

    NOTES:
        Units are arbitrary but must be self-consistent. The discretization
        yields A·phi = beta^2·phi; neff = beta / k with k = 2*pi / wavelength.
    """

    k = 2.0 * np.pi / wavelength

    boundary = boundary.upper()
    if len(boundary) != 4:
        raise ValueError("boundary string must have exactly four characters")
    if any(ch not in {"A", "S", "0"} for ch in boundary):
        raise ValueError("boundary string may contain only 'A', 'S', or '0'")

    field = field.lower()
    if field not in {"ex", "ey", "scalar"}:
        raise ValueError("field must be 'EX', 'EY', or 'scalar'")

    eps = np.asarray(eps)
    if eps.ndim != 2:
        raise ValueError("eps must be a 2-D array")

    ny, nx = eps.shape
    N = nx * ny

    eps = np.pad(eps, pad_width=1, mode="edge")

    if np.isscalar(dx):   # uniform x grid
        dx = np.full(nx + 2, dx)
    else:                 # nonuniform x grid
        dx = np.pad(np.asarray(dx), pad_width=1, mode="edge")

    if np.isscalar(dy):   # uniform y grid
        dy = np.full(ny + 2, dy)
    else:                 # nonuniform y grid
        dy = np.pad(np.asarray(dy), pad_width=1, mode="edge")

    e = np.tile((dx[1:-1] + dx[2:]) / 2, ny)
    w = np.tile((dx[1:-1] + dx[0:-2]) / 2, ny)
    p = np.tile(dx[1:-1], ny)

    n = np.repeat((dy[1:-1] + dy[2:]) / 2, nx)
    s = np.repeat((dy[1:-1] + dy[0:-2]) / 2, nx)
    q = np.repeat(dy[1:-1], nx)

    en = eps[2:ny + 2, 1:nx + 1].ravel(order="C")  # north
    es = eps[0:ny, 1:nx + 1].ravel(order="C")      # south
    ee = eps[1:ny + 1, 2:nx + 2].ravel(order="C")  # east
    ew = eps[1:ny + 1, 0:nx].ravel(order="C")      # west
    ep = eps[1:ny + 1, 1:nx + 1].ravel(order="C")  # center

    if field == "ex":
        un = 2.0 / n / (n + s)
        us = 2.0 / s / (n + s)
        ue = (
            8.0 * (p * (ep - ew) + 2.0 * w * ew) * ee
            / (
                (p * (ep - ee) + 2.0 * e * ee) * (p**2 * (ep - ew) + 4.0 * w**2 * ew)
                + (p * (ep - ew) + 2.0 * w * ew) * (p**2 * (ep - ee) + 4.0 * e**2 * ee)
            )
        )
        uw = (
            8.0 * (p * (ep - ee) + 2.0 * e * ee) * ew
            / (
                (p * (ep - ee) + 2.0 * e * ee) * (p**2 * (ep - ew) + 4.0 * w**2 * ew)
                + (p * (ep - ew) + 2.0 * w * ew) * (p**2 * (ep - ee) + 4.0 * e**2 * ee)
            )
        )
        up = ep * k**2 - un - us - ue * ep / ee - uw * ep / ew

    elif field == "ey":
        un = (
            8.0 * (q * (ep - es) + 2.0 * s * es) * en
            / (
                (q * (ep - en) + 2.0 * n * en) * (q**2 * (ep - es) + 4.0 * s**2 * es)
                + (q * (ep - es) + 2.0 * s * es) * (q**2 * (ep - en) + 4.0 * n**2 * en)
            )
        )
        us = (
            8.0 * (q * (ep - en) + 2.0 * n * en) * es
            / (
                (q * (ep - en) + 2.0 * n * en) * (q**2 * (ep - es) + 4.0 * s**2 * es)
                + (q * (ep - es) + 2.0 * s * es) * (q**2 * (ep - en) + 4.0 * n**2 * en)
            )
        )
        ue = 2.0 / e / (e + w)
        uw = 2.0 / w / (e + w)
        up = ep * k**2 - un * ep / en - us * ep / es - ue - uw

    else:  # field == "scalar"
        un = 2.0 / n / (n + s)
        us = 2.0 / s / (n + s)
        ue = 2.0 / e / (e + w)
        uw = 2.0 / w / (e + w)
        up = ep * k**2 - un - us - ue - uw

    jj = np.arange(nx * ny).reshape((ny, nx), order="C")

    # NORTH boundary
    jb = jj[ny - 1, :].ravel(order="C")
    if boundary[0] == "S":
        up[jb] += un[jb]
    elif boundary[0] == "A":
        up[jb] -= un[jb]
    # SOUTH boundary
    jb = jj[0, :].ravel(order="C")
    if boundary[1] == "S":
        up[jb] += us[jb]
    elif boundary[1] == "A":
        up[jb] -= us[jb]
    # EAST boundary
    jb = jj[:, nx - 1].ravel(order="C")
    if boundary[2] == "S":
        up[jb] += ue[jb]
    elif boundary[2] == "A":
        up[jb] -= ue[jb]
    # WEST boundary
    jb = jj[:, 0].ravel(order="C")
    if boundary[3] == "S":
        up[jb] += uw[jb]
    elif boundary[3] == "A":
        up[jb] -= uw[jb]

    jall = jj.ravel(order="C")
    jn = jj[1:ny, :].ravel(order="C")        # north neighbor indices
    js = jj[0:ny - 1, :].ravel(order="C")    # south neighbor indices
    je = jj[:, 1:nx].ravel(order="C")        # east neighbor indices
    jw = jj[:, 0:nx - 1].ravel(order="C")    # west neighbor indices

    rows = np.concatenate([jall, jw, je, js, jn])
    cols = np.concatenate([jall, je, jw, jn, js])
    data = np.concatenate([up[jall], ue[jw], uw[je], un[js], us[jn]])

    A = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    A.eliminate_zeros()

    # Solve A·phi = beta^2·phi by shift-invert. SciPy's eigs(sigma=...) factorizes
    # (A - sigma*I) with SuperLU internally -- no external sparse solver needed.
    shift = (2.0 * np.pi * guess / wavelength) ** 2  # = (k*guess)^2
    vals, vecs = eigs(A, k=nmodes, sigma=shift, which="LM", tol=1e-8)

    if np.iscomplexobj(A.data):
        phi = np.zeros((ny, nx, nmodes), dtype=complex)
    else:
        phi = np.zeros((ny, nx, nmodes), dtype=float)
        if not np.allclose(vals.imag, 0.0, 1e-12):
            warnings.warn("Unexpected complex eigenvalues, discarding imaginary parts", RuntimeWarning)
        vals = vals.real
        if not np.allclose(vecs.imag, 0.0, 1e-12):
            warnings.warn("Unexpected complex eigenvectors, discarding imaginary parts", RuntimeWarning)
        vecs = vecs.real

    # neff from eigenvalues beta^2: neff = sqrt(beta^2) / k
    neff = np.sqrt(vals) / k

    order = np.argsort(-neff.real)  # sort modes by descending effective index
    neff = neff[order]
    vecs = vecs[:, order]

    for m in range(nmodes):
        v = vecs[:, m]
        vmax = np.max(np.abs(v))
        norm = 1.0 if vmax == 0 else vmax
        phi[:, :, m] = (v / norm).reshape((ny, nx), order="C")

    if nmodes == 1:
        phi = phi[:, :, 0]
        neff = neff[0]

    return neff, phi


# Note on field labels: the semivectorial formulations 'EX'/'EY' map to the
# historical 'TM'/'TE' labels respectively -- 'EX' = E along the horizontal axis
# (the guide width here), 'EY' = E along the vertical axis. Quasi-TE for a wide,
# thin SOI core is therefore field='EX'.


# -----------------------------------------------------------------------------
# Starter-kit convenience: package a solved mode as a wavesimpro source
# -----------------------------------------------------------------------------
# For genuine 3D simulations the transverse (y, z) cross-section is a real 2D
# index map = svmodes' native input, so call ``svmodes`` directly and feed the
# resulting 2D mode profile to ``mode_source`` with ``source_plane='yz'``.


def mode_source(
    mode_profile: Sequence[complex],
    origin: str,
    position: Sequence[float],
    source_plane: str,
    pixel_size: float,
    amplitude: complex = 1.0,
):
    """Package a precomputed guided-mode profile as a wavesimpro source.

    Mirrors the (source_values, position) convention of ``wsutils.create_source``
    (e.g. ``gaussian_beam``), so the result can be passed straight to
    ``simulate(sources=[mode_source(...)])``. Handles both a 1D profile spanning a
    single transverse axis ('x'/'y'/'z') and a 2D profile spanning a transverse
    plane ('xy'/'yz'/'xz') -- the latter is the genuine 3D waveguide launch: a 2D
    mode profile on a fixed-propagation-coordinate plane.

    Args:
        mode_profile: complex mode amplitude. 1D for a single-axis ``source_plane``;
            2D for a two-axis ``source_plane`` (its two axes ordered as the sorted
            plane string, e.g. ``'yz'`` -> profile shape ``(n_y, n_z)``).
        origin: 'center' or 'topleft' -- how ``position`` locates the profile along
            the transverse (source_plane) axes.
        position: 3 floats (x, y, z) for scalar/Helmholtz, or 4 floats
            (polarization axis, x, y, z) for vector/Maxwell, in micrometer (μm).
        source_plane: the transverse axis/axes the profile spans: a single axis
            ('x'/'y'/'z') or a plane ('xy'/'yz'/'xz').
        pixel_size: grid spacing in micrometer (μm).
        amplitude: scalar to scale the source by. Default 1.0.
    Returns:
        source_values, position: same convention as the create_source helpers.
    """
    if origin not in ["center", "topleft"]:
        raise ValueError(f"Specify origin from the options ['center', 'topleft']. Invalid origin: '{origin}'")
    source_plane = "".join(sorted(source_plane.lower()))
    if source_plane not in ["x", "y", "z", "xy", "yz", "xz"]:
        raise ValueError(f"source_plane must be one of x, y, z, xy, yz, xz. Invalid: '{source_plane}'")

    profile = np.asarray(mode_profile, dtype=np.complex64) * amplitude
    if len(source_plane) != profile.ndim:
        raise ValueError(f"mode_profile.ndim ({profile.ndim}) must match the number of axes in source_plane '{source_plane}' ({len(source_plane)})")

    check_source_parameters(position, shape=profile.shape, source_plane=source_plane)
    p = polarization_(position)

    span_px = profile.shape  # samples along each plane axis, in plane-string order
    source_values = expand_source_shape(profile, source_plane)
    source_values = source_values[None, ...] if p else source_values  # add polarization dim if present

    position = np.asarray(position)
    position[p:] = position[p:] / pixel_size
    if origin == "center":
        for ax, m in zip(source_plane, span_px):
            position[p + "xyz".index(ax)] -= m // 2
    position = tuple(position.astype(int))

    return source_values.astype(np.complex64), position
