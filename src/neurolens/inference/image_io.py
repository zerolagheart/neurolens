"""Safe decoding of untrusted uploaded image bytes."""

from __future__ import annotations

import io
import warnings

from PIL import Image, UnidentifiedImageError

from neurolens.data.transforms import load_image_rgb

SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF"}
MAX_PIXELS = 40_000_000  # decompression-bomb guard (~6300 x 6300)
MIN_SIDE = 32


class InvalidImageError(ValueError):
    """The upload is not a decodable image in a supported format."""


class ImageTooLargeError(InvalidImageError):
    """The upload exceeds the byte or pixel limit."""


class UnsupportedFormatError(InvalidImageError):
    """The upload decodes, but in a format the service does not accept."""


def decode_image(data: bytes, max_bytes: int | None = None) -> Image.Image:
    """Validate and decode ``data`` into an RGB PIL image.

    Raises:
        InvalidImageError: empty, too large, unsupported format, corrupt or implausibly small.
    """
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    if max_bytes is not None and len(data) > max_bytes:
        raise ImageTooLargeError(
            f"The uploaded file is {len(data) / 1e6:.1f} MB; the limit is {max_bytes / 1e6:.1f} MB."
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                fmt = probe.format
                width, height = probe.size
                probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise InvalidImageError("The file could not be decoded as an image.") from exc
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ImageTooLargeError("The image dimensions are too large.") from exc
    if fmt not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"Unsupported image format '{fmt}'. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}."
        )
    if width * height > MAX_PIXELS:
        raise ImageTooLargeError("The image dimensions are too large.")
    if min(width, height) < MIN_SIDE:
        raise InvalidImageError(
            f"The image is too small ({width}x{height}); minimum side is {MIN_SIDE}px."
        )
    try:
        return load_image_rgb(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("The image data is truncated or corrupt.") from exc
