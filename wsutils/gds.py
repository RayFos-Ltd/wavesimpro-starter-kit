"""Write polygons to a GDSII layout file, with the standard library only."""

import math
import struct

import numpy as np

_DB_UNIT_UM = 1e-4  # coordinates are stored as whole multiples of 0.1 nm
_MAX_POINTS = 8190  # most vertices one GDSII polygon record holds


def _real8(value: float) -> bytes:
    """A positive number as a GDSII 8-byte real: base-16 exponent, 56-bit mantissa."""
    exponent = 64
    while value >= 1:
        value /= 16
        exponent += 1
    while value < 1 / 16:
        value *= 16
        exponent -= 1
    return bytes([exponent]) + int(round(value * 2 ** 56)).to_bytes(7, "big")


def _record(record_type: int, data_type: int, payload: bytes = b"") -> bytes:
    return struct.pack(">HBB", 4 + len(payload), record_type, data_type) + payload


def _name(text: str) -> bytes:
    data = text.encode("ascii")
    return data + (b"\0" if len(data) % 2 else b"")


def circle(center, radius: float, segments: int = 48) -> np.ndarray:
    """A regular polygon with the same area as the circle, as ``(segments, 2)`` vertices in um."""
    r = radius * math.sqrt(2 * math.pi / (segments * math.sin(2 * math.pi / segments)))
    angle = 2 * math.pi * np.arange(segments) / segments
    return np.column_stack([center[0] + r * np.cos(angle), center[1] + r * np.sin(angle)])


def write_gds(path, polygons, layer: int = 1, name: str = "TOP") -> None:
    """Write polygons to one layer of a GDSII file.

    Args:
        path: file to write.
        polygons: sequence of ``(n, 2)`` arrays of (x, y) vertices in um, each one outline
            without the closing point.
        layer: GDS layer number of every polygon.
        name: name of the library and of its single structure.
    """
    stamp = struct.pack(">12h", *([2000, 1, 1, 0, 0, 0] * 2))
    out = [_record(0x00, 2, struct.pack(">h", 600)),  # HEADER, stream version 6
           _record(0x01, 2, stamp),  # BGNLIB
           _record(0x02, 6, _name(name)),  # LIBNAME
           _record(0x03, 5, _real8(_DB_UNIT_UM) + _real8(_DB_UNIT_UM * 1e-6)),  # UNITS
           _record(0x05, 2, stamp),  # BGNSTR
           _record(0x06, 6, _name(name))]  # STRNAME
    for polygon in polygons:
        points = np.rint(np.asarray(polygon, dtype=float) / _DB_UNIT_UM).astype(np.int64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError(f"a polygon must be an (n, 2) array with n >= 3, got shape {points.shape}")
        if len(points) > _MAX_POINTS:
            raise ValueError(f"a GDSII polygon holds at most {_MAX_POINTS} vertices, got {len(points)}")
        closed = np.vstack([points, points[:1]])
        out += [_record(0x08, 0),  # BOUNDARY
                _record(0x0D, 2, struct.pack(">h", layer)),  # LAYER
                _record(0x0E, 2, struct.pack(">h", 0)),  # DATATYPE
                _record(0x10, 3, closed.astype(">i4").tobytes()),  # XY
                _record(0x11, 0)]  # ENDEL
    out += [_record(0x07, 0), _record(0x04, 0)]  # ENDSTR, ENDLIB
    with open(path, "wb") as file:
        file.write(b"".join(out))
