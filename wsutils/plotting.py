import os

import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from time import localtime, strftime

from .utilities import normalize, relative_error

# Common folder for figures/artifacts produced by the examples. Git-ignored
# (see .gitignore) so generated output never gets committed.
OUTPUT_DIR = "output"


def _save_figure(filename):
    """Save the current figure as PNG + PDF, creating the parent dir if needed."""
    parent = os.path.dirname(filename)
    if parent:
        os.makedirs(parent, exist_ok=True)
    plt.savefig(f'{filename}.png', bbox_inches='tight', pad_inches=0.03, dpi=300)
    plt.savefig(f'{filename}.pdf', bbox_inches='tight', pad_inches=0.03, dpi=300)


def _to_plane(u):
    """Reduce a field to a 2D (x, y) plane for imaging.

    4D (polarization, x, y, z) -> x-polarization; 3D (x, y, z) -> XY slice at the middle of z.
    Returns (plane, slice index, number of z slices); the last two are None for 2D input.
    """
    if u.ndim == 4:
        u = u[0, ...]  # x-polarization
    if u.ndim == 3:
        k = u.shape[2] // 2
        return u[:, :, k], k, u.shape[2]
    return u, None, None


def _extent(shape, pixel_size, centered):
    """[left, right, bottom, top] in µm for an (x, y) array, y pointing up (as in the web GUI)."""
    nx, ny = shape
    if centered:
        return np.array([-nx / 2, nx / 2, -ny / 2, ny / 2]) * pixel_size
    return np.array([0, nx, 0, ny]) * pixel_size


def _slice_title(k, n, pixel_size, centered):
    """Title naming the XY slice shown, e.g. for matching the slice slider in the web GUI."""
    if k is None:
        return None
    z = (k + 0.5 - (n / 2 if centered else 0)) * pixel_size
    return rf"XY slice, $z$ = {z:.3g} $\mu m$ (index {k}/{n})"


def _imshow_field(u, pixel_size, centered=True, axis_labels=("x", "y"), cmap="inferno", log=False, vmin=None, vmax=None):
    """Show a 2D (x, y) array with x horizontal and y pointing up, matching the web GUI."""
    if log:
        kwargs = dict(norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    else:
        kwargs = dict(vmin=vmin, vmax=vmax)
    plt.imshow(u.T, origin="lower", cmap=cmap, extent=_extent(u.shape, pixel_size, centered), **kwargs)
    plt.xlabel(rf"${axis_labels[0]}~(\mu m)$")
    plt.ylabel(rf"${axis_labels[1]}~(\mu m)$")


def plot_computed(u, pixel_size, normalize_u=True, cmap="inferno", log=False, show=True, save=False, filename=None,
                  centered=True, axis_labels=("x", "y")):
    """Plot a computed field. 1D: real and imaginary parts. 2D/3D/4D: |u| of the (central XY) plane.

    Images follow the web GUI convention: y points up and, with centered=True, the axes are
    centred on the domain (0 at the middle), matching origin='center' in create_medium/create_source.
    axis_labels names the two plotted axes, e.g. ("x", "z") for an XZ monitor plane.
    """
    date_time = strftime("%Y%m%d_%H%M%S", localtime())
    if u.ndim == 1:
        u_range = np.arange(u.shape[0]) * pixel_size
        plt.subplot(211)
        plt.plot(u_range, u.real)
        plt.xlabel(r"$x~(\mu m)$")
        plt.title("Real part")
        plt.grid()

        plt.subplot(212)
        plt.plot(u_range, u.imag)
        plt.xlabel(r"$x~(\mu m)$")
        plt.title("Imaginary part")
        plt.grid()

        plt.tight_layout()
        if filename is None:
            filename = f'{OUTPUT_DIR}/fig1d_{date_time}'

    else:
        u, k, n = _to_plane(u)
        u = np.abs(u)

        if normalize_u:
            min_val = np.min(u)
            u = normalize(u, a=1e-10 if min_val == 0 else 0.01 * min_val)

        _imshow_field(u, pixel_size, centered, axis_labels, cmap, log)
        title = _slice_title(k, n, pixel_size, centered)
        if title:
            plt.title(title)
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.set_title(r"$|E|$")
        plt.tight_layout()

        if filename is None:
            filename = f'{OUTPUT_DIR}/fig2d_{date_time}'

    if save:
        _save_figure(filename)
    plt.show() if show else None
    plt.close('all')


def plot_computed_and_reference(u, u_ref, pixel_size, re=None, normalize_u=True, cmap="inferno", log=False, show=True, save=False, filename=None,
                                centered=True, axis_labels=("x", "y")):
    """Plot the computed field u and the reference field u_ref.
    If u and u_ref are 1D arrays, the real and imaginary parts are plotted separately.
    If u and u_ref are 2D arrays, the absolute values are plotted.
    If u and u_ref are 3D arrays, the central XY slice (middle of z) is plotted; 4D uses the x-polarization.
    Images follow the web GUI convention (y up; centred axes if centered=True), as in plot_computed.
    If normalize_u is True, the values are normalized to the same range.
    The relative error is (computed, if needed, and) displayed.
    """

    re = relative_error(u, u_ref) if re is None else re
    u_range = np.arange(u.shape[0]) * pixel_size
    date_time = strftime("%Y%m%d_%H%M%S", localtime())

    if u.ndim == 1 and u_ref.ndim == 1:
        plt.subplot(211)
        plt.plot(u_range, u_ref.real, label="Reference")
        plt.plot(u_range, u.real, label="Computed")
        plt.plot(u_range, u_ref.real - u.real, label="Difference")
        plt.xlabel(r"$x~(\mu m)$")
        plt.legend()
        plt.title(f"Real part (RE = {relative_error(u.real, u_ref.real):.2e})")
        plt.grid()

        plt.subplot(212)
        plt.plot(u_range, u_ref.imag, label="Reference")
        plt.plot(u_range, u.imag, label="Computed")
        plt.plot(u_range, u_ref.imag - u.imag, label="Difference")
        plt.xlabel(r"$x~(\mu m)$")
        plt.legend()
        plt.title(f"Imaginary part (RE = {relative_error(u.imag, u_ref.imag):.2e})")
        plt.grid()

        plt.suptitle(f"Relative error (RE) = {re:.2e}")
        plt.tight_layout()

        if filename is None:
            filename = f'logs/fig1d_{date_time}'

    else:
        u, k, n = _to_plane(u)
        u_ref, _, _ = _to_plane(u_ref)

        u = np.abs(u)
        u_ref = np.abs(u_ref)
        if normalize_u:
            min_val = min(np.min(u), np.min(u_ref))
            max_val = max(np.max(u), np.max(u_ref))
            a = 1e-10 if min_val == 0 else 0.01 * min_val
            b = 1
            u = normalize(u, min_val, max_val, a, b)
            u_ref = normalize(u_ref, min_val, max_val, a, b)
        else:
            a = None
            b = None

        plt.figure(figsize=(10, 5))
        for i, (field, name) in enumerate([(u_ref, "Reference"), (u, "Computed")]):
            plt.subplot(121 + i)
            _imshow_field(field, pixel_size, centered, axis_labels, cmap, log, vmin=a, vmax=b)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(name)

        slice_title = _slice_title(k, n, pixel_size, centered)
        plt.suptitle(f"Relative error (RE) = {re:.2e}" + (f" — {slice_title}" if slice_title else ""))
        plt.tight_layout()
        if filename is None:
            filename = f'{OUTPUT_DIR}/fig2d_{date_time}'

    if save:
        _save_figure(filename)
    plt.show() if show else None
    plt.close('all')