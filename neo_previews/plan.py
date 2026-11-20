"""Decide, per LoRA in a folder, whether to skip, extract a thumbnail, or generate."""

import os
from collections.abc import Callable
from dataclasses import dataclass

from . import prompt as prompts
from . import scan, thumbnail

SKIP = "skip"
EXTRACT = "extract"
GENERATE = "generate"
# Would generate, but Forge has not listed the file (a hidden folder with
# list_hidden_files off, for example), so <lora:...> could not load it and
# the image would be made without the LoRA.
UNLISTED = "unlisted"

GENERATED_EXTENSION = "png"


@dataclass
class Item:
    path: str
    action: str
    existing: str | None = None  # the preview Forge shows now, for SKIP
    thumbnail: tuple[bytes, str] | None = None  # for EXTRACT
    target: str | None = None  # the preview file to write
    prompt: prompts.Prompt | None = None  # for GENERATE
    full_prompt: str = ""

    @property
    def stem(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]


def default_alias(path: str, metadata: dict) -> str:
    """Forge's default ("Alias from file"): ss_output_name, else the file name.

    Inside Forge the caller passes NetworkOnDisk.get_alias() instead, which
    also follows the lora_preferred_name setting and alias collisions.
    """
    return metadata.get("ss_output_name") or os.path.splitext(os.path.basename(path))[0]


def plan_one(
    path: str,
    templates: prompts.Templates,
    *,
    overwrite: bool = False,
    use_thumbnails: bool = True,
    alias: Callable[[str, dict], str | None] = default_alias,
    main_prompt: str = "",
    main_negative: str = "",
) -> Item:
    """alias returns the name for <lora:NAME:...>, or None when Forge cannot load the file.

    main_prompt / main_negative: the txt2img prompt boxes; see prompt.with_main_prompt.
    """
    metadata = scan.read_metadata(path)
    existing = scan.existing_preview(path, metadata)
    if existing and not overwrite:
        return Item(path, SKIP, existing=existing)

    if use_thumbnails:
        found = thumbnail.embedded_thumbnail(metadata)
        if found:
            return Item(path, EXTRACT, existing=existing, thumbnail=found, target=thumbnail.target_path(path, found[1]))

    chosen = prompts.with_main_prompt(prompts.choose(path, metadata, templates), main_prompt, main_negative)
    name = alias(path, metadata)
    if name is None:
        return Item(path, UNLISTED, existing=existing, prompt=chosen)
    full = prompts.lora_prompt(name, chosen.weight, chosen.text)
    target = thumbnail.target_path(path, GENERATED_EXTENSION)
    return Item(path, GENERATE, existing=existing, target=target, prompt=chosen, full_prompt=full)


def plan_folder(folder: str, templates: prompts.Templates, **options) -> list[Item]:
    return [plan_one(path, templates, **options) for path in scan.walk(folder)]


def displaced_previews(item: Item) -> list[str]:
    """Preview files that would hide the new one when overwriting.

    find_preview takes the first match in its extension order, so an older
    <name>.png stays on the card in front of a new <name>.preview.jpg.
    The target itself is replaced, not listed.
    """
    if not item.target:
        return []
    target = os.path.normcase(os.path.abspath(item.target))
    return [p for p in scan.preview_files(item.path) if os.path.normcase(os.path.abspath(p)) != target]


def set_aside(paths: list[str]) -> list[str]:
    """Rename older previews to <file>.old so the new one shows. Returns the new names."""
    moved = []
    for path in paths:
        old = path + ".old"
        os.replace(path, old)
        moved.append(old)
    return moved


def write_preview(item: Item, data: bytes) -> list[str]:
    """Write a preview for item. Older previews that would hide it are set aside first."""
    moved = set_aside(displaced_previews(item))
    thumbnail.write_preview_bytes(item.target, data)
    return moved
