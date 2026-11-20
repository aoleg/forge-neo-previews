"""Walk a LoRA folder, read safetensors headers, and apply Forge's preview rule."""

import json
import os
import re
import struct

# Files Forge lists on the Lora page (networks.process_network_files).
LORA_EXTENSIONS = (".safetensors", ".pt", ".ckpt")

# modules/ui_extra_networks.py default_allowed_preview_extensions, in the
# order find_preview tries them.
PREVIEW_EXTENSIONS = ("png", "jpg", "jpeg", "webp", "jxl", "avif", "heif", "gif", "mp4", "webm")

MAX_HEADER = 100 * 1024 * 1024


def natural_sort_key(text: str, _digits=re.compile(r"([0-9]+)")) -> list:
    """Same order as Forge's modules/util.py natural_sort_key."""
    return [[int(t) if t.isdigit() else t.lower() for t in _digits.split(part)] for part in os.path.splitext(text)]


def is_hidden(name: str) -> bool:
    return name.startswith(".")


def walk(folder: str) -> list[str]:
    """LoRA files under folder, recursively, in Forge's order.

    Folders whose name starts with "." are skipped below `folder`. The folder
    itself may be hidden: the user picked it on purpose.
    """
    found = []
    for root, dirs, files in os.walk(folder, followlinks=True):
        dirs[:] = [d for d in dirs if not is_hidden(d)]
        for name in files:
            if os.path.splitext(name)[1].lower() in LORA_EXTENSIONS:
                found.append(os.path.join(root, name))
    return sorted(found, key=lambda p: [natural_sort_key(part) for part in os.path.relpath(p, folder).split(os.sep)])


def folder_choices(roots: list[str], *, show_hidden: bool = False) -> dict[str, str]:
    """Label -> folder, for every folder that holds a LoRA in it or below it.

    A label starts with the root's own name ("Lora/krea/test"), with " (2)"
    and so on when two roots share a name. Folders starting with "." are left
    out unless show_hidden (Forge's list_hidden_files): Forge does not list
    the LoRAs in them, so it could not apply one.
    """
    choices: dict[str, str] = {}
    seen_roots: set[str] = set()
    used_names: dict[str, int] = {}
    for root in roots:
        if not root:
            continue
        root = os.path.abspath(str(root))
        key = os.path.normcase(root)
        if key in seen_roots or not os.path.isdir(root):
            continue
        seen_roots.add(key)
        name = os.path.basename(root.rstrip("\\/")) or root
        used_names[name] = used_names.get(name, 0) + 1
        if used_names[name] > 1:
            name = f"{name} ({used_names[name]})"

        with_lora: set[str] = set()
        for directory, dirs, files in os.walk(root, followlinks=True):
            if not show_hidden:
                dirs[:] = [d for d in dirs if not is_hidden(d)]
            if any(os.path.splitext(f)[1].lower() in LORA_EXTENSIONS for f in files):
                rel = os.path.relpath(directory, root)
                parts = [] if rel == "." else rel.split(os.sep)
                for depth in range(len(parts) + 1):
                    with_lora.add("/".join(parts[:depth]))
        for rel in sorted(with_lora, key=lambda r: [natural_sort_key(p) for p in r.split("/")] if r else []):
            label = f"{name}/{rel}" if rel else name
            choices[label] = os.path.join(root, *rel.split("/")) if rel else root
    return choices


def read_metadata(path: str) -> dict:
    """The safetensors __metadata__ (string values), or {} for other files and on any error."""
    if not path.lower().endswith(".safetensors"):
        return {}
    try:
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            if size > MAX_HEADER:
                return {}
            header = json.loads(f.read(size))
    except (OSError, ValueError, struct.error):
        return {}
    metadata = header.get("__metadata__") if isinstance(header, dict) else None
    return metadata if isinstance(metadata, dict) else {}


def base_path(path: str) -> str:
    """The LoRA path without its extension, as Forge uses it for sidecar files."""
    return os.path.splitext(path)[0]


def preview_files(path: str) -> list[str]:
    """Existing preview files of a LoRA, in the order find_preview tries them."""
    base = base_path(path)
    found = []
    for ext in PREVIEW_EXTENSIONS:
        for candidate in (f"{base}.{ext}", f"{base}.preview.{ext}"):
            if os.path.isfile(candidate):
                found.append(candidate)
    return found


def has_cover_images(path: str, metadata: dict) -> bool:
    """find_embedded_preview: a non-empty ssmd_cover_images in a .safetensors file."""
    if not path.lower().endswith(".safetensors"):
        return False
    raw = metadata.get("ssmd_cover_images")
    if not raw:
        return False
    try:
        covers = json.loads(raw)
    except (TypeError, ValueError):
        return False
    return isinstance(covers, list) and any(covers)


def existing_preview(path: str, metadata: dict) -> str | None:
    """What Forge shows on the card: a preview file, the embedded cover, or None."""
    files = preview_files(path)
    if files:
        return files[0]
    if has_cover_images(path, metadata):
        return "ssmd_cover_images"
    return None
