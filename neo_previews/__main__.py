"""Dry run: python -m neo_previews <folder> [options]

Lists every LoRA under the folder with what the extension would do:
skip (has a preview), extract (embedded thumbnail) or generate (with the
prompt shown). Writes nothing unless --extract is given.
"""

import argparse
import os
import sys

from . import plan, prompt


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neo_previews", description="List what forge-neo-previews would do for each LoRA in a folder.")
    parser.add_argument("folder", help="LoRA folder, worked recursively (folders starting with '.' below it are skipped)")
    parser.add_argument("--all", action="store_true", help="also list LoRAs that already have a preview")
    parser.add_argument("--overwrite", action="store_true", help="plan as if existing previews were to be replaced")
    parser.add_argument("--no-thumbnails", action="store_true", help="never extract embedded thumbnails")
    parser.add_argument("--extract", action="store_true", help="write the embedded thumbnails now (no GPU needed)")
    parser.add_argument("--weight", type=float, default=1.0, help="weight when the LoRA has no preferred weight (default 1)")
    parser.add_argument("--trigger-template", default=prompt.Templates.trigger)
    parser.add_argument("--sample-template", default=prompt.Templates.sample)
    parser.add_argument("--name-template", default=prompt.Templates.name)
    parser.add_argument("--negative-template", default=prompt.Templates.negative)
    parser.add_argument("--prompt", default="", help="as the main prompt box: used for every LoRA, {lora} becomes the LoRA's own prompt")
    parser.add_argument("--negative", default="", help="as the main negative prompt box: {lora} becomes the LoRA's own negative")
    parser.add_argument("--width", type=int, default=160, help="cut prompts to this many characters (0 = never)")
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        print(f"Not a folder: {folder}", file=sys.stderr)
        return 2

    templates = prompt.Templates(trigger=args.trigger_template, sample=args.sample_template, name=args.name_template, negative=args.negative_template, default_weight=args.weight)
    items = plan.plan_folder(folder, templates, overwrite=args.overwrite, use_thumbnails=not args.no_thumbnails, main_prompt=args.prompt, main_negative=args.negative)

    counts: dict[str, int] = {}
    sources: dict[str, int] = {}
    for item in items:
        counts[item.action] = counts.get(item.action, 0) + 1
        if item.action == plan.GENERATE:
            sources[item.prompt.source] = sources.get(item.prompt.source, 0) + 1
        if item.action == plan.SKIP and not args.all:
            continue

        rel = os.path.relpath(item.path, folder)
        print(f"[{item.action}] {rel}")
        if item.action == plan.SKIP:
            shown = item.existing if item.existing == "ssmd_cover_images" else os.path.basename(item.existing)
            print(f"    preview: {shown}")
        elif item.action == plan.EXTRACT:
            data, ext = item.thumbnail
            print(f"    thumbnail: {ext}, {len(data):,} bytes -> {os.path.basename(item.target)}")
        else:
            cut = (lambda s: _shorten(s, args.width)) if args.width else (lambda s: s)
            print(f"    source: {item.prompt.source}")
            print(f"    prompt: {cut(item.full_prompt)}")
            if item.prompt.negative:
                print(f"    negative: {cut(item.prompt.negative)}")
        for old in plan.displaced_previews(item) if args.overwrite and item.action != plan.SKIP else ():
            print(f"    sets aside: {os.path.basename(old)}")

        if args.extract and item.action == plan.EXTRACT:
            moved = plan.write_preview(item, item.thumbnail[0])
            print(f"    written: {os.path.basename(item.target)}" + (f" (set aside {len(moved)})" if moved else ""))

    print()
    print(f"{len(items)} LoRA files: " + ", ".join(f"{counts.get(a, 0)} {a}" for a in (plan.SKIP, plan.EXTRACT, plan.GENERATE)))
    if sources:
        print("prompt sources: " + ", ".join(f"{n} {s}" for s, n in sorted(sources.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
