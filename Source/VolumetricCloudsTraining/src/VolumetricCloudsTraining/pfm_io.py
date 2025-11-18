import re
import struct
from typing import Tuple, Union

import numpy as np


# ================================================================
#                       PFM I/O UTILITIES
# ================================================================
# Portable Float Map reader/writer supporting both color ("PF")
# and grayscale ("Pf") formats. Follows the de facto standard:
# - Header: "PF" or "Pf"
# - Next line: "<width> <height>"
# - Next line: <scale>; negative => little endian, positive => big endian
# - Binary data: float32, rows stored from top to bottom in the file,
#   but many producers historically store bottom-to-top. We'll follow
#   the common convention of flipping vertically on read and write so
#   arrays use row 0 = top.
# ================================================================


def _parse_dimensions(token: str) -> Tuple[int, int]:
    parts = token.strip().split()
    if len(parts) != 2:
        raise ValueError(f"PFM header expects two integers for dimensions, got: {token!r}")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"PFM dimensions must be positive, got: {(width, height)}")
    return width, height


def read_pfm(path: str) -> np.ndarray:
    """
    Read a PFM file into a NumPy array of shape (H, W) for grayscale or (H, W, 3) for color.
    Returned dtype is float32. Rows are top-to-bottom.
    """
    with open(path, "rb") as f:
        header = f.readline().decode("ascii").rstrip()
        if header not in ("PF", "Pf"):
            raise ValueError(f"Not a PFM file (header {header!r}) at {path}")

        dim_line = f.readline().decode("ascii")
        # Some writers include comments; robustly skip comment lines
        while dim_line.startswith("#"):
            dim_line = f.readline().decode("ascii")
        width, height = _parse_dimensions(dim_line)

        scale_line = f.readline().decode("ascii").strip()
        while scale_line.startswith("#"):
            scale_line = f.readline().decode("ascii").strip()

        scale = float(scale_line)
        # Negative scale => little endian; positive => big endian
        little_endian = scale < 0
        endian_char = "<" if little_endian else ">"
        num_channels = 3 if header == "PF" else 1

        num_floats = width * height * num_channels
        data = f.read(num_floats * 4)
        if len(data) != num_floats * 4:
            raise ValueError(f"Unexpected EOF while reading PFM data in {path}")

        array = np.frombuffer(data, dtype=endian_char + "f4")
        array = np.reshape(array, (height, width, num_channels)) if num_channels == 3 else np.reshape(array, (height, width))

        # Flip vertically so index 0 is the top row
        array = np.flipud(array)

        # Standardize dtype
        return array.astype(np.float32, copy=False)


def write_pfm(path: str, image: Union[np.ndarray, "np.typing.NDArray[np.float32]"], scale: float = -1.0) -> None:
    """
    Write a NumPy array to a PFM file.
    - image shape: (H, W) for grayscale or (H, W, 3) for color
    - dtype: float32 recommended
    - scale: negative for little endian (typical), positive for big endian
    Rows are written bottom-to-top per PFM convention.
    """
    if image.ndim not in (2, 3):
        raise ValueError(f"PFM expects 2D (grayscale) or 3D (color) array, got shape {image.shape}")

    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise ValueError(f"Color PFM expects last dim 1 or 3, got {image.shape[2]}")

    if image.dtype != np.float32:
        image = image.astype(np.float32)

    height, width = image.shape[:2]
    color = image.ndim == 3 and image.shape[2] == 3
    header = "PF" if color else "Pf"

    # Flip vertically for PFM write
    image_to_write = np.flipud(image)

    # Determine endianness from scale sign
    little_endian = scale < 0
    endian_char = "<" if little_endian else ">"

    with open(path, "wb") as f:
        f.write(f"{header}\n".encode("ascii"))
        f.write(f"{width} {height}\n".encode("ascii"))
        f.write(f"{scale}\n".encode("ascii"))
        f.write(image_to_write.astype(endian_char + "f4", copy=False).tobytes())


