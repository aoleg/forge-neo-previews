"""The SAI model-spec thumbnail (modelspec.thumbnail) that Forge does not show."""

import base64
import binascii
import os
import re

from . import scan

_DATA_URL = re.compile(r"\s*data:image/([a-zA-Z0-9.+-]+)\s*;\s*base64\s*,(.*)", re.DOTALL)

# What the bytes say, checked before the declared type.
_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"\xff\x0a", "jxl"),
    (b"\x00\x00\x00\x0cJXL \r\n\x87\n", "jxl"),
)

_DECLARED = {"jpeg": "jpg", "jpg": "jpg", "pjpeg": "jpg", "png": "png", "webp": "webp", "gif": "gif", "avif": "avif", "jxl": "jxl", "heif": "heif", "heic": "heif"}


def sniff(data: bytes) -> str | None:
    """Image format from the leading bytes, as a Forge preview extension."""
    for signature, ext in _SIGNATURES:
        if data.startswith(signature):
            return ext
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"avif", b"avis"):
            return "avif"
        if brand in (b"heic", b"heix", b"mif1", b"msf1"):
            return "heif"
    return None


def embedded_thumbnail(metadata: dict) -> tuple[bytes, str] | None:
    """(image bytes, extension) from modelspec.thumbnail, or None.

    The bytes are kept as they are: original size and format. None when the
    value is not a base64 image data URL or the format is not one Forge shows.
    """
    value = metadata.get("modelspec.thumbnail")
    if not isinstance(value, str):
        return None
    match = _DATA_URL.match(value)
    if not match:
        return None
    try:
        data = base64.b64decode("".join(match.group(2).split()), validate=True)
    except (binascii.Error, ValueError):
        return None
    if not data:
        return None
    ext = sniff(data) or _DECLARED.get(match.group(1).lower())
    if ext not in scan.PREVIEW_EXTENSIONS:
        return None
    return data, ext


def target_path(lora_path: str, ext: str) -> str:
    return f"{scan.base_path(lora_path)}.preview.{ext}"


def write_preview_bytes(target: str, data: bytes) -> None:
    """Write through a temporary file, so a half-written preview never shows."""
    temp = f"{target}.tmp"
    with open(temp, "wb") as f:
        f.write(data)
    os.replace(temp, target)
