"""
Maxwell 3D inverted-pyramid light trapping — NIR absorption in a CMOS pixel surface
====================================================================================
Reproduces the unit-cell FDTD study of Yokogawa et al., "IR sensitivity enhancement
of CMOS Image Sensor with diffractive light trapping pixels", Sci. Rep. 7:3832 (2017),
DOI 10.1038/s41598-017-04200-y — a CMOS image sensor with enhanced IR sensitivity.

A crystalline-Si photodiode is nearly transparent at NIR wavelengths (at λ = 850 nm
the absorption depth is ~19 μm, so a 3 μm photodiode catches only ~12%). The paper's fix:
etch a dense array of inverted pyramids (IPA) into the Si surface — the (111) facets
at 54.7° diffract the incoming light into steep angles inside the Si, elongating the
optical path and trapping light by total internal reflection. That is pure wavelength-
scale diffraction: no ray model captures it, and the pyramid size (~λ/2) is exactly
the regime where effective-medium (moth-eye) approximations fail too. Full-wave 3D
Maxwell in an ABSORBING medium is the only honest tool.

This script solves ONE periodic unit cell of the IPA (paper Fig. 2 setup): SiO2 above,
semi-infinite c-Si below (emulated by an absorbing bottom boundary), plane wave at
normal incidence, and integrates the absorbed power over the top 3 μm of Si.

Validation anchors from the paper (λ = 850 nm, stated in the text):
    flat surface     A = 12.5 %
    IPA pitch 400 nm A = 16.5 %   (+31 %)
    IPA pitch 540 nm A = 18.1 %   (FOM 1.43, the optimum at 850 nm)
The flat case is ALSO checked against the analytic single-pass Fresnel answer, which
this geometry (semi-infinite Si -> no bottom reflection) should match exactly.

-------------------------------------------------------------------------------
GRID PITCH & COST — this is the Modified Born Series (MBS) solver, NOT FDTD
-------------------------------------------------------------------------------
The paper needed a 2 nm Lumerical FDTD mesh to resolve the pyramid facets. MBS
converges at λ/4–λ/8 IN THE MEDIUM: at λ = 850 nm in Si (n ≈ 3.65) that is ~30 nm.
We use 20 nm — a factor 10 coarser per axis, i.e. ~1000x fewer voxels per volume,
with no staircase-driven mesh refinement. One unit cell here is ~30x30x260 voxels
(~0.2 Mvoxel): each solve is seconds, so the pitch/wavelength maps of the paper
(Figs. 2a–c) are cheap to fill in once this validation passes.

Physics/geometry notes:
  * Inverted pyramid pit: square mouth = one full pitch at the Si surface (densely
    packed, zero spacing), apex pointing down, depth = (pitch/2)·tan(54.7°) ≈ 0.71·pitch
    (the Si(111)/(100) facet angle fixes the aspect ratio — it is not a free knob).
    Pits are refilled with SiO2 (the paper planarizes with passivation/oxide).
  * Absorbing Si: complex index n - already used for the metal mask in
    maxwell_microlens.py; absorbed power density is (ω ε0 / 2)·Im(ε)·|E|².
    Absorbed fraction A = k0·Σ Im(ε)|E|²·dV / (n_SiO2·|E0|²·A_cell), with |E0|²
    measured by an identical reference run in uniform SiO2 (same trick as the
    reference run in the GMR spectrum example).
  * Normal incidence on a square-symmetric cell: one linear polarization (E_x)
    is sufficient by symmetry. (For oblique incidence, average two polarizations.)
  * CLOUD DATA TRANSFER: the solve runs remotely, so the returned field is the
    main per-run cost. With USE_MONITORS=True only three regions come back over
    the wire — the absorbing Si block (what the metric needs), one thin x-z
    slice for the field figure, and a single reference plane for |E0|² — instead
    of the full (3, nx, ny, nz) volume. That is what makes many-run sweeps
    (wavelengths x pitches) cheap to iterate. Set USE_MONITORS=False to fall
    back to full-field return (useful as a cross-check of the monitor path).
  * Start SMALL: one wavelength (850 nm), three cases. Extend WAVELENGTHS and
    PITCHES below once the anchors are reproduced — each extra point is one cheap
    solve, and n(λ), k(λ) are interpolated from the embedded Green-2008 table.

Run from the repo root (see CLAUDE.md):
    python examples/maxwell_ipa_light_trapping.py
Outputs land in the git-ignored output/ folder.
"""

import os
import signal
from time import time

import numpy as np
import matplotlib.pyplot as plt

from wavesimpro import configure, create_monitor, simulate
from wsutils import OUTPUT_DIR, plane_wave

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# Parameters
# =============================================================================
# DEFAULT: the paper's three text-stated anchor points at 850 nm — 4 solves, a few
# minutes, and the console tells you directly whether you reproduced the paper
# (expected: flat 12.1% [analytic 12.07%], 400 nm 15.8% FOM 1.31 [paper 1.32],
# 540 nm 17.9% FOM 1.48 [paper 1.43]).
WAVELENGTHS = [0.850]
PITCHES = [None, 0.400, 0.540]
# EXTENSION — full reproduction of the paper's Fig. 2c FOM-vs-pitch curves
# (30 solves, ~25 min; the 1000 nm row uses 10λ absorbers, see boundary_widths,
# and carries ±0.05-0.1 FOM near its Rayleigh-threshold pitches 400/800 nm):
# WAVELENGTHS = [0.700, 0.850, 1.000]
# PITCHES = [None, 0.200, 0.300, 0.400, 0.500, 0.540, 0.600, 0.700, 0.800]

pixel_size = 0.020    # μm. λ/8 in Si at λ=700 nm is ~23 nm, so 20 nm covers the whole
                      # 700–1000 nm band; a 10 nm convergence check reproduces the 20 nm
                      # results. (The paper's FDTD needed a 2 nm mesh for this geometry —
                      # that contrast is the MBS story.)
n_sio2 = 1.45         # SiO2 buffer above the Si and pyramid backfill (paper value)
flat_cell = 0.400     # μm, transverse cell for the flat case (size is arbitrary under
                      # periodic BCs; matching the 400 nm case reuses its reference run)

# --- z-stack (propagation axis; x, y are the periodic in-plane axes) ----------
z_src = 0.30          # plane-wave source plane, inside the SiO2 (μm)
z_si0 = 1.20          # Si surface = pyramid mouths (μm)
si_depth_integr = 3.0  # absorption integrated over the top 3 μm of Si (paper's metric)
si_depth_total = 8.0   # Si continues past the metric depth into the absorbing bottom
                       # boundary -> emulates SEMI-INFINITE Si. Generous margin + a wide
                       # absorber matter here: the IPA diffracts into GRAZING orders in
                       # the Si (up to ~75° off-normal), where absorbing boundaries
                       # reflect worst — any bounce-back recrosses the 3 μm metric
                       # volume and inflates A for the IPA cases (not the flat one).
def boundary_widths(wavelength):
    """Absorbing-layer thickness (μm), specified in VACUUM WAVELENGTHS so a sweep
    cannot silently fall below the 4-5λ MBS sizing rule at its red end.

    The demanding case is the BOTTOM absorber: the IPA diffracts into steep orders
    that hit it far off-normal, where any absorbing ramp partially reflects. At
    700/850 nm the Si itself damps those orders before they matter and 5λ is
    sufficient. At 1000 nm the Si is nearly transparent (absorption depth ~156 μm)
    so the orders arrive undamped; a 5λ absorber then still leaks and the
    Rayleigh-threshold pitches (≈ λ/n_Si and 2λ/n_Si) read spuriously high FOMs.
    10λ gives converged values matching the paper's Fig. 2c curve. General rule:
    confirm any near-threshold point by re-running with a thicker boundary."""
    return (10.0 if wavelength >= 0.95 else 5.0) * wavelength
z_end = z_si0 + si_depth_total
nz = int(round(z_end / pixel_size))
iz_si0 = int(round(z_si0 / pixel_size))
iz_3um = int(round((z_si0 + si_depth_integr) / pixel_size))
iz_e0 = (int(round(2.0 / pixel_size)), int(round(4.0 / pixel_size)))  # |E0|² slab (full-field path)

facet_deg = 54.7      # Si(111) facet angle to the (100) surface — fixed by crystallography

# --- Monitors: ship only what we use back from the cloud ----------------------
# Monitor coords are domain-centred world coords: voxel i on an n-voxel axis sits
# at (i + 0.5 - n/2)*pixel_size. z-extents use ODD voxel counts so the returned
# block (2*floor(N/2)+1 voxels for a size of N*pixel_size) aligns exactly with
# the requested grid range.
USE_MONITORS = True
# Extents are DERIVED from physical lengths (not hardcoded voxel counts) so a
# pixel_size refinement run keeps the same physical monitor regions.
M_SI = 2 * (int(round((si_depth_integr + 0.6) / pixel_size)) // 2) + 1
                      # z voxels of returned Si block: surface down ~3.6 μm
                      # (covers the 3 μm metric + margin for the depth-profile plot)
IZ_SLICE0 = int(round(0.8 / pixel_size))          # x-z viz slice: 0.8 μm ..
_m = nz - 1 - IZ_SLICE0                           # .. to one voxel short of z_end
M_SLICE = _m if _m % 2 else _m - 1
IZ_REF = int(round(3.0 / pixel_size))  # single |E0|² plane (reference run, monitor path)
assert iz_si0 + M_SI <= nz and iz_3um - iz_si0 <= M_SI

# Paper anchors at λ = 850 nm (absorbed fraction in the top 3 μm of Si)
PAPER_A_850 = {None: 0.125, 0.400: 0.165, 0.540: 0.181}
# Optimal IPA pitch per wavelength, as STATED IN THE PAPER'S TEXT (Fig. 2c
# discussion): "the optimal pitch of 440 nm, 540 nm and 640 nm at each wavelength".
PAPER_OPT_PITCH = {0.700: 0.440, 0.850: 0.540, 1.000: 0.640}


# =============================================================================
# Dispersion: complex refractive index of c-Si, 0.70–1.00 μm
# =============================================================================
# Tabulated from M. A. Green, Sol. Energy Mater. Sol. Cells 92, 1305 (2008)
# ("Self-consistent optical parameters of intrinsic silicon at 300 K").
# k = α·λ/4π from the tabulated absorption coefficient. Swap in your own
# measured (λ, n, k) table via np.interp exactly the same way.
_SI_TABLE = np.array([
    # λ (μm)   n       k
    [0.70, 3.780, 1.06e-2],
    [0.75, 3.732, 7.58e-3],
    [0.80, 3.694, 5.41e-3],
    [0.85, 3.653, 3.62e-3],
    [0.90, 3.629, 2.19e-3],
    [0.95, 3.607, 1.19e-3],
    [1.00, 3.588, 5.1e-4],
])


def n_si(wavelength):
    """Complex refractive index n + ik of c-Si at the given wavelength (μm).
    Positive imaginary part = absorbing, same convention as the lossy metal
    mask (n = 1 + 3j) in maxwell_microlens.py."""
    n = np.interp(wavelength, _SI_TABLE[:, 0], _SI_TABLE[:, 1])
    k = np.interp(wavelength, _SI_TABLE[:, 0], _SI_TABLE[:, 2])
    return complex(n, k)


def fresnel_flat_absorption(wavelength):
    """Analytic check for the FLAT case: single-pass absorption in the top 3 μm
    of semi-infinite Si behind the SiO2/Si Fresnel interface. No bottom
    reflection (semi-infinite), so this is exact up to O(k²) ~ 1e-5."""
    nc = n_si(wavelength)
    r = (n_sio2 - nc) / (n_sio2 + nc)
    T = 1.0 - abs(r) ** 2
    alpha = 4 * np.pi * nc.imag / wavelength     # intensity absorption coeff (1/μm)
    return T * (1.0 - np.exp(-alpha * si_depth_integr))


# =============================================================================
# Medium: one periodic unit cell — SiO2 | inverted pyramid | bulk Si
# =============================================================================
def build_cell(pitch, wavelength):
    """Permittivity (n²) of one periodic cell, shape (nxy, nxy, nz) complex64.
    pitch=None -> flat Si surface (reference).

    Monitors truncate ODD transverse voxel counts by one, which would
    corrupt the absorption integral. So the cell is forced to an EVEN count:
    if pitch/pixel_size is odd, the cell holds 2x2 pyramids instead of one —
    physically identical under periodic boundaries.

    Returns (permittivity, nxy, iy_slice) where iy_slice is the y-row through
    the centre (apex) of the first pyramid, for the field figure."""
    if pitch is None:
        nxy = int(round(flat_cell / pixel_size))
        n = np.full((nxy, nxy, nz), n_sio2, dtype=np.complex64)
        n[:, :, iz_si0:] = n_si(wavelength)
        return (n ** 2).astype(np.complex64), nxy, nxy // 2

    npp = int(round(pitch / pixel_size))     # voxels per pitch
    reps = 1 if npp % 2 == 0 else 2          # 2x2 pyramids if one pitch is odd
    nxy = reps * npp
    n = np.full((nxy, nxy, nz), n_sio2, dtype=np.complex64)
    n[:, :, iz_si0:] = n_si(wavelength)

    # Inverted pyramid(s): square mouth = full pitch at the surface, apex down.
    # cheb = square (Chebyshev) distance to the NEAREST pyramid centre, so the
    # same code tiles 1 or 2 pyramids per side. At depth t the etched
    # (SiO2-refilled) square has half-width w(t).
    depth = pitch / 2 * np.tan(np.deg2rad(facet_deg))
    xy = (np.arange(nxy) + 0.5) * pixel_size
    d = np.abs(xy - (np.floor(xy / pitch) + 0.5) * pitch)
    cheb = np.maximum(d[:, None], d[None, :])
    for iz in range(iz_si0, min(nz, iz_si0 + int(np.ceil(depth / pixel_size)))):
        t = (iz - iz_si0 + 0.5) * pixel_size
        w = pitch / 2 * (1 - t / depth)
        if w <= 0:
            break
        n[cheb < w, iz] = n_sio2
    return (n ** 2).astype(np.complex64), nxy, npp // 2


# =============================================================================
# One solve: full-cell plane wave, periodic in x & y, absorbing in z
# =============================================================================
def run(permittivity, nxy, wavelength, monitors=None):
    # Full-cell uniform sheet source + periodic BCs = clean plane wave. The
    # (nxy + 0.5) guards plane_wave's internal μm->px truncation against float
    # round-off (e.g. 0.6/0.02 -> 29.999...).
    source_values, source_position = plane_wave(
        shape=[(nxy + 0.5) * pixel_size, (nxy + 0.5) * pixel_size],
        origin="topleft",
        position=[0, 0.0, 0.0, z_src],   # polarization axis 0 = E_x (transverse)
        source_plane="xy",
        pixel_size=pixel_size,
    )
    u, iterations, residual = simulate(
        permittivity=permittivity,
        sources=[(source_values, source_position)],
        wavelength=wavelength,
        pixel_size=pixel_size,
        boundary_width=boundary_widths(wavelength),  # λ-scaled absorbers, z only
        periodic=(True, True, False),    # bottom absorber extends Si -> semi-infinite
        use_gpu=True,
        monitors=monitors,               # None -> full field; list -> LIST of blocks
    )
    return u, iterations, residual


def zcenter(iz0, m):
    """Domain-centred world z of a monitor spanning voxels iz0 .. iz0+m-1."""
    return (iz0 + (m - 1) / 2 + 0.5 - nz / 2) * pixel_size


def make_monitors(nxy, iy_slice):
    """Structure-run monitors: (1) the Si block the absorption metric needs,
    (2) one voxel-thin x-z slice through a pyramid apex for the field figure."""
    full = nxy * pixel_size
    si_block = create_monitor(
        name="si_block", center_x=0.0, center_y=0.0,
        center_z=zcenter(iz_si0, M_SI),
        size_x=full, size_y=full, size_z=M_SI * pixel_size)
    xz_slice = create_monitor(
        name="xz_slice", center_x=0.0,
        center_y=(iy_slice + 0.5 - nxy / 2) * pixel_size,
        center_z=zcenter(IZ_SLICE0, M_SLICE),
        size_x=full, size_y=pixel_size, size_z=M_SLICE * pixel_size)
    return [si_block, xz_slice]


def run_structure(permittivity, nxy, iy_slice, wavelength):
    """Solve one cell. Returns (esq_si, mag_slice, iterations, residual):
    esq_si = |E|² (pol-summed) over the Si block, shape (nxy, nxy, M_SI);
    mag_slice = |E| on the apex x-z slice, shape (nxy, M_SLICE)."""
    if USE_MONITORS:
        u, iterations, residual = run(permittivity, nxy, wavelength,
                                      monitors=make_monitors(nxy, iy_slice))
        u_si, u_slice = u[0], u[1]
    else:
        u, iterations, residual = run(permittivity, nxy, wavelength)
        u_si = u[:, :, :, iz_si0:iz_si0 + M_SI]
        u_slice = u[:, :, iy_slice:iy_slice + 1, IZ_SLICE0:IZ_SLICE0 + M_SLICE]
    esq_si = (np.abs(u_si) ** 2).sum(axis=0)
    assert esq_si.shape == (nxy, nxy, M_SI), \
        f"Si monitor returned {esq_si.shape}, expected {(nxy, nxy, M_SI)} — " \
        "the absorption sum must cover the full cell cross-section"
    mag_slice = np.sqrt((np.abs(u_slice) ** 2).sum(axis=0))[:, 0, :]
    return esq_si, mag_slice.astype(np.float32), iterations, residual


def run_reference(nxy, wavelength):
    """Uniform-SiO2 run calibrating the incident plane-wave |E0|² per voxel
    (source amplitude -> field amplitude is not 1:1)."""
    eps_ref = np.full((nxy, nxy, nz), n_sio2 ** 2, dtype=np.complex64)
    if USE_MONITORS:
        ref_plane = create_monitor(
            name="e0_plane", center_x=0.0, center_y=0.0,
            center_z=zcenter(IZ_REF, 1),
            size_x=nxy * pixel_size, size_y=nxy * pixel_size, size_z=pixel_size)
        u, _, _ = run(eps_ref, nxy, wavelength, monitors=[ref_plane])
        return float((np.abs(u[0]) ** 2).sum(axis=0).mean())
    u, _, _ = run(eps_ref, nxy, wavelength)
    return float((np.abs(u) ** 2).sum(axis=0)[:, :, iz_e0[0]:iz_e0[1]].mean())


# =============================================================================
# Sweep: for each wavelength, a reference (uniform SiO2) run + one run per pitch
# =============================================================================
configure(description="maxwell_ipa_light_trapping")

results = []       # dicts: wavelength, pitch, A, fom
profiles = {}      # (wavelength, pitch) -> cumulative absorption vs depth, for plots
fields = {}        # (wavelength, pitch) -> |E| x-z slice through the cell centre

for wavelength in WAVELENGTHS:
    k0 = 2 * np.pi / wavelength
    nc = n_si(wavelength)
    print(f"\nlambda = {wavelength*1e3:.0f} nm   n_Si = {nc.real:.3f} + {nc.imag:.2e}j "
          f"(absorption depth {wavelength / (4 * np.pi * nc.imag):.1f} um)")

    # ONE |E0|² calibration per wavelength: a uniform sheet source under periodic
    # BCs radiates a fixed amplitude PER UNIT AREA, so the reference cannot depend
    # on the transverse cell size — never re-run it per pitch/cell.
    e0sq = run_reference(int(round(flat_cell / pixel_size)), wavelength)
    a_flat = None

    for pitch in PITCHES:
        permittivity, nxy, iy_slice = build_cell(pitch, wavelength)

        label = "flat" if pitch is None else f"pitch {pitch*1e3:.0f} nm"
        start = time()
        esq_si, mag_slice, iters, res = run_structure(permittivity, nxy, iy_slice, wavelength)

        # Absorbed fraction: A = k0 * sum(Im(eps)|E|^2) * dV / (n_SiO2 |E0|^2 A_cell),
        # with dV/A_cell = pixel_size / (nxy*nxy). Restricted to the top 3 um of Si
        # (SiO2 has Im(eps)=0, so the pyramid backfill contributes nothing).
        eps_im = permittivity.imag[:, :, iz_si0:iz_si0 + M_SI]
        prof = (eps_im * esq_si).sum(axis=(0, 1))   # per z-slice of the Si block
        norm = k0 * pixel_size / (n_sio2 * e0sq * nxy * nxy)
        A = norm * prof[:iz_3um - iz_si0].sum()

        if pitch is None:
            a_flat = A
        fom = A / a_flat if a_flat else float("nan")
        results.append(dict(wavelength=wavelength, pitch=pitch, A=A, fom=fom))
        profiles[(wavelength, pitch)] = norm * np.cumsum(prof)
        fields[(wavelength, pitch)] = mag_slice

        anchor = PAPER_A_850.get(pitch) if abs(wavelength - 0.850) < 1e-6 else None
        anchor_txt = f"  paper {anchor*100:.1f}%" if anchor else ""
        print(f"  {label:<13} A = {A*100:5.2f}%  FOM = {fom:4.2f}{anchor_txt}   "
              f"({nxy}x{nxy}x{nz} vox, {iters} iters, res {res:.1e}, {time()-start:.1f} s)")

    print(f"  analytic Fresnel check (flat): A = {fresnel_flat_absorption(wavelength)*100:.2f}% "
          f"vs simulated {a_flat*100:.2f}%")


# =============================================================================
# Restore SIGINT before plotting (cloud client can leave it wedged)
# =============================================================================
try:
    signal.signal(signal.SIGINT, signal.default_int_handler)
except (ValueError, TypeError):
    pass


# =============================================================================
# Figure 1: |E| cross-sections (flat vs best IPA) + cumulative absorption depth
# =============================================================================
wl0 = 0.850 if 0.850 in WAVELENGTHS else WAVELENGTHS[0]   # headline wavelength
show_pitches = [None, 0.540 if 0.540 in PITCHES else PITCHES[-1]]
profile_pitches = [p for p in PITCHES if p in (None, 0.400, 0.540, 0.800)]
vmax = max(fields[(wl0, p)].max() for p in show_pitches)
span_x = 1.7  # μm of tiled periods to display (the periodic field tiles exactly)

fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1, 1, 1.4]})
z_slice0, z_slice1 = IZ_SLICE0 * pixel_size, (IZ_SLICE0 + M_SLICE) * pixel_size
for ax, pitch in zip(axes[:2], show_pitches):
    mag = fields[(wl0, pitch)]                       # (nxy, M_SLICE) slice at y = centre
    cell = mag.shape[0] * pixel_size
    tiles = int(np.ceil(span_x / cell))
    mag = np.tile(mag, (tiles, 1))
    extent = [0, tiles * cell, z_slice1, z_slice0]   # depth increases downward
    ax.imshow(mag.T, cmap="inferno", extent=extent, vmin=0, vmax=vmax, aspect="auto")
    ax.axhline(z_si0, color="cyan", lw=0.8, ls="--", alpha=0.8)
    ax.axhline(z_si0 + si_depth_integr, color="lime", lw=0.8, ls=":", alpha=0.8)
    ax.set_ylim(z_slice1 - 0.2, z_slice0)            # crop source run-up clutter
    ax.set_xlabel(r"$x~(\mu m)$")
    ax.set_ylabel(r"$z~(\mu m)$ — depth")
    ax.set_title("flat surface" if pitch is None
                 else f"IPA, pitch {pitch*1e3:.0f} nm")

ax = axes[2]
z_si = (np.arange(M_SI) + 0.5) * pixel_size
for pitch in profile_pitches:
    lab = "flat" if pitch is None else f"IPA {pitch*1e3:.0f} nm"
    ax.plot(z_si, profiles[(wl0, pitch)] * 100, label=lab)
ax.axvline(si_depth_integr, color="gray", lw=0.8, ls="--")
ax.text(si_depth_integr, ax.get_ylim()[0], "  3 μm metric depth",
        rotation=90, va="bottom", fontsize=8, color="gray")
ax.set_xlabel("depth into Si (μm)")
ax.set_ylabel("cumulative absorbed fraction (%)")
ax.set_title("light trapping = steeper absorption\nper unit depth (path elongation)")
ax.grid(True, alpha=0.3)
ax.legend()
fig.suptitle(f"Inverted-pyramid light trapping in c-Si, λ = {wl0*1e3:.0f} nm  "
             "(cyan = Si surface, green = 3 μm metric depth)", fontsize=11)
fig.savefig(f"{OUTPUT_DIR}/maxwell_ipa_fields.png", dpi=300, bbox_inches="tight")


# =============================================================================
# Figure 2: paper Fig. 2c — absorption & FOM vs pitch, one curve per wavelength.
# The FOM optimum tracks pitch ≈ λ/n_SiO2 (Wood's anomaly), dashed verticals.
# =============================================================================
fig2, (axA, axF) = plt.subplots(1, 2, figsize=(12.5, 5))
for wavelength in WAVELENGTHS:
    rows = [r for r in results if r["wavelength"] == wavelength]
    x = [0 if r["pitch"] is None else r["pitch"] * 1e3 for r in rows]
    line, = axA.plot(x, [r["A"] * 100 for r in rows], "o-", ms=4,
                     label=f"λ = {wavelength*1e3:.0f} nm")
    axF.plot(x, [r["fom"] for r in rows], "o-", ms=4, color=line.get_color(),
             label=f"λ = {wavelength*1e3:.0f} nm")
    axF.axvline(wavelength / n_sio2 * 1e3, ls="--", lw=0.9,
                color=line.get_color(), alpha=0.5)
    opt = PAPER_OPT_PITCH.get(round(wavelength, 3))
    if opt is not None:   # paper's text-stated optimum pitch for this wavelength
        axF.plot(opt * 1e3, 1.0, marker="^", ms=10, ls="none",
                 color=line.get_color(), clip_on=False)
paper_x = [0 if p is None else p * 1e3 for p in PAPER_A_850]
axA.plot(paper_x, [v * 100 for v in PAPER_A_850.values()], "s", mfc="none",
         mec="k", ms=9, label="Yokogawa 2017 @ 850 nm\n(FDTD, 2 nm mesh)")
axF.plot(paper_x, [v / PAPER_A_850[None] for v in PAPER_A_850.values()], "s",
         mfc="none", mec="k", ms=9, label="Yokogawa 2017 @ 850 nm")
axF.plot([], [], marker="^", ls="none", color="k",
         label="paper optimum pitch (440/540/640 nm)")
axA.set_xlabel("IPA pitch (nm)   [0 = flat surface]")
axA.set_ylabel("absorption in top 3 μm of c-Si (%)")
axA.set_title("absolute absorption")
axF.set_xlabel("IPA pitch (nm)   [0 = flat surface]")
axF.set_ylabel("FOM = A(pitch) / A(flat)")
axF.set_title("enhancement (paper Fig. 2c)\n"
              "dashed: pitch = λ/n$_{SiO_2}$ (Wood's anomaly)")
for ax in (axA, axF):
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
fig2.suptitle("IPA light trapping vs Yokogawa 2017 — "
              "same physics, ~1000× fewer voxels than the paper's FDTD", fontsize=11)
fig2.tight_layout()
fig2.savefig(f"{OUTPUT_DIR}/maxwell_ipa_absorption.png", dpi=300, bbox_inches="tight")

np.savetxt(f"{OUTPUT_DIR}/maxwell_ipa_absorption.csv",
           np.array([[r["wavelength"] * 1e3, 0 if r["pitch"] is None else r["pitch"] * 1e3,
                      r["A"], r["fom"]] for r in results]),
           delimiter=",", header="wavelength_nm,pitch_nm,absorbed_fraction,fom", comments="")

print(f"\nSaved {OUTPUT_DIR}/maxwell_ipa_fields.png, maxwell_ipa_absorption.png and .csv")
plt.show()
