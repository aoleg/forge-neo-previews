"""Text and HTML for the dry run and the end-of-run summary."""

import html
import os
from dataclasses import dataclass, field

from . import plan

ACTIONS = (plan.EXTRACT, plan.GENERATE, plan.UNLISTED, plan.SKIP)
LABELS = {
    plan.EXTRACT: "extract",
    plan.GENERATE: "generate",
    plan.UNLISTED: "not listed by Forge",
    plan.SKIP: "has a preview",
}
LATER = "generate, next run"


def counts(items: list[plan.Item]) -> dict[str, int]:
    result = dict.fromkeys(ACTIONS, 0)
    for item in items:
        result[item.action] = result.get(item.action, 0) + 1
    return result


def count_line(items: list[plan.Item]) -> str:
    c = counts(items)
    parts = [f"{c[a]} {LABELS[a]}" for a in ACTIONS if c[a] or a != plan.UNLISTED]
    return f"{len(items)} LoRA files: " + ", ".join(parts)


def limited(items: list[plan.Item], max_count: int) -> tuple[list[plan.Item], list[plan.Item]]:
    """The GENERATE items of this run, and those left for a later one."""
    generate = [i for i in items if i.action == plan.GENERATE]
    if max_count and max_count > 0:
        return generate[:max_count], generate[max_count:]
    return generate, []


def _rel(path: str, folder: str) -> str:
    """Path below the folder with "/" separators, like the folder labels."""
    try:
        return os.path.relpath(path, folder).replace(os.sep, "/")
    except ValueError:
        return path


def html_table(items: list[plan.Item], folder: str, max_count: int = 0, limit: int = 500) -> str:
    """The dry run: one row per LoRA that would get a preview, newest decisions first."""
    _, later = limited(items, max_count)
    later_ids = {id(i) for i in later}
    rows = []
    shown = [i for i in items if i.action != plan.SKIP]
    for item in shown[:limit]:
        action = LATER if id(item) in later_ids else LABELS[item.action]
        if item.action == plan.EXTRACT:
            data, ext = item.thumbnail
            detail = html.escape(f"embedded thumbnail, {ext}, {len(data):,} bytes")
            source = "modelspec.thumbnail"
        elif item.action == plan.GENERATE:
            detail = html.escape(item.full_prompt)
            if item.prompt.negative:
                detail += f"<br><span class='npv-neg'>negative: {html.escape(item.prompt.negative)}</span>"
            source = item.prompt.source
        else:
            detail = html.escape("Forge has not listed this file (hidden folder?), so it cannot be applied")
            source = item.prompt.source if item.prompt else ""
        rows.append(f"<tr class='npv-{item.action}'><td>{html.escape(action)}</td><td>{html.escape(_rel(item.path, folder))}</td><td>{html.escape(source)}</td><td>{detail}</td></tr>")
    more = len(shown) - len(rows)
    head = f"<p>{html.escape(count_line(items))}" + (f"; {len(later)} left for a later run (maximum {max_count})" if later else "") + "</p>"
    if not rows:
        return head + "<p>Nothing to do: every LoRA here has a preview.</p>"
    table = "<table class='npv-table'><thead><tr><th>Action</th><th>File</th><th>Source</th><th>Prompt</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    return head + table + (f"<p>… and {more} more.</p>" if more > 0 else "")


@dataclass
class RunResult:
    """What one run did, for the summary."""

    extracted: list[plan.Item] = field(default_factory=list)
    generated: list[plan.Item] = field(default_factory=list)
    not_applied: list[tuple[plan.Item, str]] = field(default_factory=list)  # the LoRA did not load or did not fit the model
    skipped: list[plan.Item] = field(default_factory=list)  # the user pressed Skip
    failed: list[tuple[plan.Item, str]] = field(default_factory=list)  # writing the file, or no image came back
    not_reached: int = 0  # planned for this run, but the run stopped first
    later: int = 0  # beyond the maximum per run
    stopped: str = ""  # "interrupted" or "stopped after the current image"


def summary(folder: str, items: list[plan.Item], result: RunResult) -> str:
    """Plain text for the txt2img info box and the console."""
    c = counts(items)

    def files(entries, attr="path"):
        return [f"  {_rel(getattr(e, attr), folder)}" for e in entries]

    def reasons(entries):
        return [f"  {_rel(e.path, folder)}: {reason}" for e, reason in entries]

    lines = [f"LoRA previews: {folder}"]
    if c[plan.EXTRACT] or result.extracted:
        lines.append(f"Extracted embedded thumbnails: {len(result.extracted)}" + (f" of {c[plan.EXTRACT]}" if len(result.extracted) != c[plan.EXTRACT] else ""))
        lines += files(result.extracted, "target")
    lines.append(f"Generated previews: {len(result.generated)}")
    lines += files(result.generated, "target")
    if result.not_applied:
        lines.append(f"LoRA not applied, nothing saved: {len(result.not_applied)}")
        lines += reasons(result.not_applied)
    if result.skipped:
        lines.append(f"Skipped, nothing saved: {len(result.skipped)}")
        lines += files(result.skipped)
    if result.failed:
        lines.append(f"Failed: {len(result.failed)}")
        lines += reasons(result.failed)
    if result.not_reached:
        lines.append(f"Not reached: {result.not_reached}")
    if result.later:
        lines.append(f"Left for a later run (maximum per run): {result.later}")
    if c[plan.UNLISTED]:
        lines.append(f"Not listed by Forge, left alone: {c[plan.UNLISTED]}")
    lines.append(f"Already had a preview: {c[plan.SKIP]}")
    if result.stopped:
        lines.append(result.stopped[0].upper() + result.stopped[1:] + ".")
    if result.generated or result.extracted:
        lines.append("Press Refresh on the Lora page to see the new previews.")
    return "\n".join(lines)
