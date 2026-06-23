"""
Procedural PIC layout primitives
================================
Engine-free helpers for painting photonic-integrated-circuit (PIC) device
geometry as a per-voxel refractive-index map, built directly with NumPy. Used by
the waveguide-device examples (grating coupler, MMI splitter, directional
coupler) instead of importing GDS, so every dimension stays a tunable top-level
variable and the geometry rescales by editing a number.

All shapes are painted onto a 2D index *canvas* whose two axes are
``axis 0 = propagation`` and ``axis 1 = transverse`` (the guide width for a
top-down layout, or the layer-stack/thickness direction for a side view). The
physical meaning of the transverse axis is the caller's choice; these helpers
only paint. Coordinates and sizes are in micrometer (μm); ``index`` is the
refractive index ``n`` (not ``n**2``).

Finish with ``permittivity_2d`` (adds a singleton third axis and squares to
``n**2``) for a thin / periodic-third-axis 2D ``simulate`` run.
"""

import numpy as np
from typing import Sequence


def index_canvas(sim_size: Sequence[float], pixel_size: float, background: complex = 1.0):
    """Create a blank (nx, ny) refractive-index canvas filled with ``background``.

    Args:
        sim_size: (x, y) canvas size in micrometer (μm).
        pixel_size: grid spacing in micrometer (μm).
        background: background refractive index (e.g. cladding). Default 1.0 (air).
    Returns:
        n: (nx, ny) complex64 refractive-index array.
    """
    nx, ny = (np.asarray(sim_size) / pixel_size).astype(int)
    return np.full((int(nx), int(ny)), background, dtype=np.complex64)


def _px(value: float, pixel_size: float) -> int:
    return int(round(value / pixel_size))


def add_rect(n, pixel_size, x_span, y_span, index):
    """Paint an axis-aligned rectangle ``[x0,x1] x [y0,y1]`` (μm) with ``index``."""
    x0, x1 = sorted(x_span)
    y0, y1 = sorted(y_span)
    i0, i1 = max(_px(x0, pixel_size), 0), min(_px(x1, pixel_size), n.shape[0])
    j0, j1 = max(_px(y0, pixel_size), 0), min(_px(y1, pixel_size), n.shape[1])
    n[i0:i1, j0:j1] = index
    return n


def add_waveguide(n, pixel_size, x_span, y_center, width, index):
    """Paint a straight waveguide of ``width`` centered at ``y_center`` over ``x_span``."""
    return add_rect(n, pixel_size, x_span, (y_center - width / 2, y_center + width / 2), index)


def add_taper(n, pixel_size, x_span, y_center, width_start, width_end, index):
    """Paint a linear-width taper from ``width_start`` to ``width_end`` along x,
    centered on ``y_center``."""
    x0, x1 = x_span
    i0, i1 = _px(x0, pixel_size), _px(x1, pixel_size)
    for i in range(max(i0, 0), min(i1, n.shape[0])):
        t = (i - i0) / max(i1 - i0, 1)
        w = width_start + t * (width_end - width_start)
        j0 = max(_px(y_center - w / 2, pixel_size), 0)
        j1 = min(_px(y_center + w / 2, pixel_size), n.shape[1])
        n[i, j0:j1] = index
    return n


def add_s_bend(n, pixel_size, x_span, y_start, y_end, width, index):
    """Paint a raised-cosine S-bend of constant ``width`` whose centerline moves
    from ``y_start`` to ``y_end`` over ``x_span`` (smooth, zero-slope at both ends)."""
    x0, x1 = x_span
    i0, i1 = _px(x0, pixel_size), _px(x1, pixel_size)
    for i in range(max(i0, 0), min(i1, n.shape[0])):
        t = (i - i0) / max(i1 - i0, 1)
        yc = y_start + (y_end - y_start) * (0.5 - 0.5 * np.cos(np.pi * t))
        j0 = max(_px(yc - width / 2, pixel_size), 0)
        j1 = min(_px(yc + width / 2, pixel_size), n.shape[1])
        n[i, j0:j1] = index
    return n


def add_grating(n, pixel_size, x_start, period, duty, n_teeth, y_span, index):
    """Paint ``n_teeth`` periodic teeth (each ``duty*period`` wide, pitch ``period``)
    starting at ``x_start``, spanning ``y_span`` in the transverse axis.

    Used to etch a grating by painting the *tooth* (or, with a cladding ``index``,
    the etched gaps) over a previously-painted slab.
    """
    for k in range(int(n_teeth)):
        xa = x_start + k * period
        xb = xa + duty * period
        add_rect(n, pixel_size, (xa, xb), y_span, index)
    return n


def permittivity_2d(n):
    """Convert an (nx, ny) refractive-index canvas to an (nx, ny, 1) permittivity
    (``n**2``) array for a thin / periodic-third-axis 2D ``simulate`` run."""
    return (np.asarray(n, dtype=np.complex64) ** 2)[..., None]
