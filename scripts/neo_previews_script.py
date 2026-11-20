"""forge-neo-previews: "LoRA previews" in the txt2img Script dropdown.

For every LoRA without a preview in the chosen folder (recursively): an
embedded modelspec.thumbnail is written out as <name>.preview.<ext>;
otherwise a preview is generated with the current txt2img settings.

Settings: Settings > LoRA Previews.
"""

import copy
import html
import os

import gradio as gr
from modules import images, processing, script_callbacks, scripts, shared
from modules.processing import Processed
from modules.shared import state
from modules.ui_common import refresh_symbol
from modules.ui_components import ToolButton

from neo_previews import plan, prompt, report, scan

TITLE = "LoRA previews"
SECTION = ("neo_previews", "LoRA Previews")

OPTION_TRIGGER = "npv_trigger_template"
OPTION_SAMPLE = "npv_sample_template"
OPTION_NAME = "npv_name_template"
OPTION_NEGATIVE = "npv_negative_template"
OPTION_STOPWORDS = "npv_stopwords"
OPTION_SAVE_OUTPUTS = "npv_save_outputs"

DEFAULTS = prompt.Templates()

_folders: dict[str, str] = {}


def log(message: str) -> None:
    print(f"[LoRA previews] {message}")


# ---------------------------------------------------------------- settings


def on_ui_settings() -> None:
    from modules.options import OptionInfo

    options = {
        OPTION_TRIGGER: OptionInfo(DEFAULTS.trigger, "Prompt when a trigger is found").info(
            "{trigger}: activation text, Civitai trainedWords, trigger key in the file, or the most frequent training tag ; {name}: cleaned file name"
        ),
        OPTION_SAMPLE: OptionInfo(DEFAULTS.sample, "Prompt when the file carries a sample prompt").info("{sample}: modelspec.description, when it reads like a prompt"),
        OPTION_NAME: OptionInfo(DEFAULTS.name, "Prompt when nothing else is found").info("{name}: the file name without numbers, versions and the stopwords below"),
        OPTION_NEGATIVE: OptionInfo(DEFAULTS.negative, "Negative prompt").info("used when the LoRA's own metadata has no negative text ; {name}, {trigger}, {sample} work here too"),
        OPTION_STOPWORDS: OptionInfo(DEFAULTS.stopwords, "Words dropped from file names", gr.Textbox, {"lines": 3}).info("separated by spaces or commas ; case does not matter"),
        OPTION_SAVE_OUTPUTS: OptionInfo(False, "Also save generated previews to the outputs folder"),
    }
    for key, info in options.items():
        info.section = SECTION
        if key not in shared.opts.data_labels:
            shared.opts.add_option(key, info)


script_callbacks.on_ui_settings(on_ui_settings)


def templates_from_opts() -> prompt.Templates:
    opts = shared.opts

    def get(key, default):
        value = getattr(opts, key, None)
        return default if value is None else str(value)

    try:
        weight = float(getattr(opts, "extra_networks_default_multiplier", 1.0) or 1.0)
    except (TypeError, ValueError):
        weight = 1.0
    return prompt.Templates(
        trigger=get(OPTION_TRIGGER, DEFAULTS.trigger),
        sample=get(OPTION_SAMPLE, DEFAULTS.sample),
        name=get(OPTION_NAME, DEFAULTS.name),
        negative=get(OPTION_NEGATIVE, DEFAULTS.negative),
        stopwords=get(OPTION_STOPWORDS, DEFAULTS.stopwords),
        default_weight=weight,
    )


# ---------------------------------------------------------------- folders and LoRAs


def lora_roots() -> list[str]:
    return [shared.cmd_opts.lora_dir, *(shared.cmd_opts.lora_dirs or [])]


def refresh_folders() -> list[str]:
    global _folders
    _folders = scan.folder_choices(lora_roots(), show_hidden=bool(getattr(shared.opts, "list_hidden_files", False)))
    return list(_folders)


def resolve_folder(label) -> str | None:
    if not label:
        return None
    if label not in _folders:
        refresh_folders()
    return _folders.get(label)


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def forge_alias(paths: list[str]):
    """alias(path, metadata) for plan_one: Forge's own alias, or None when Forge has not listed the file."""
    import networks

    by_path = {_key(entry.filename): entry for entry in list(networks.available_networks.values())}
    missing = [p for p in paths if _key(p) not in by_path]
    if missing:
        # Downloaded since the last Refresh: add them without clearing the list a running job may use.
        networks.update_available_networks_by_names([os.path.splitext(os.path.basename(p))[0] for p in missing])
        by_path = {_key(entry.filename): entry for entry in list(networks.available_networks.values())}

    def alias(path: str, metadata: dict):
        entry = by_path.get(_key(path))
        return entry.get_alias() if entry is not None else None

    return alias


def make_plan(folder: str, overwrite: bool, use_thumbnails: bool) -> list[plan.Item]:
    paths = scan.walk(folder)
    alias = forge_alias(paths)
    templates = templates_from_opts()
    return [plan.plan_one(path, templates, overwrite=bool(overwrite), use_thumbnails=bool(use_thumbnails), alias=alias) for path in paths]


def _max_count(value) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def dry_run(label, overwrite, use_thumbnails, max_count) -> str:
    folder = resolve_folder(label)
    if folder is None:
        return "<p>Choose a LoRA folder first.</p>"
    try:
        items = make_plan(folder, overwrite, use_thumbnails)
    except Exception as e:  # noqa: BLE001 - shown in the panel, traceback in the console
        import traceback

        log("dry run failed:\n" + traceback.format_exc())
        return f"<p>Dry run failed: {html.escape(repr(e))}</p>"
    return report.html_table(items, folder, _max_count(max_count))


# ---------------------------------------------------------------- generation


_abort_pending = False


def on_after_cfg(params) -> None:
    """Stop sampling after the first step when the LoRA did not load (see LoadRecorder).

    Not earlier: an interrupted sampler returns state.current_latent, which is
    only stored at the end of a step (sd_samplers_cfg_denoiser.py, right after
    this callback). A skip before that gives Forge None and it crashes.
    """
    if _abort_pending:
        state.skipped = True


script_callbacks.on_cfg_after_cfg(on_after_cfg, name="forge-neo-previews abort")


class LoadRecorder:
    """Records, per LoRA file, whether loading it patched the model.

    Forge's networks.load_lora_for_models answers an architecture mismatch
    with a log warning and returns the model unpatched; the LoRA never shows
    up as failed. While a run is active the function is wrapped (it is looked
    up as a module global at call time), and put back afterwards.

    When the file being previewed did not patch anything, an abort is armed:
    on_after_cfg stops sampling after one step instead of making a full image
    without the LoRA.
    """

    def __init__(self):
        self.applied: dict[str, bool] = {}
        self.expected = ""

    def __enter__(self):
        import networks

        self.networks = networks
        self.original = networks.load_lora_for_models
        networks.load_lora_for_models = self.wrapper
        return self

    def __exit__(self, *exc):
        global _abort_pending
        _abort_pending = False
        self.networks.load_lora_for_models = self.original
        return False

    def wrapper(self, model, clip, *args, **kwargs):
        new_model, new_clip = self.original(model, clip, *args, **kwargs)
        filename = kwargs.get("filename", args[3] if len(args) > 3 else "")
        key = _key(filename) if filename else ""
        dynamic_args = getattr(self.networks, "dynamic_args", None)
        # Nunchaku models take the LoRA in their own way and always hand back the same objects.
        applied = new_model is not model or new_clip is not clip or bool(getattr(dynamic_args, "nunchaku", False))
        self.applied[key] = self.applied.get(key, False) or applied
        if not applied and key == self.expected:
            global _abort_pending
            _abort_pending = True
        return new_model, new_clip

    def expect(self, path: str) -> None:
        global _abort_pending
        _abort_pending = False
        self.expected = _key(path)
        self.applied.pop(self.expected, None)

    def result(self, path: str):
        """True, False, or None when Forge never tried to load it."""
        return self.applied.get(_key(path))


def failed_load_reason(item: plan.Item) -> str:
    """Forge's own error for a LoRA it could not read, if any."""
    import networks

    name = item.full_prompt.split(":", 2)[1] if item.full_prompt.startswith("<lora:") else item.stem
    for key in (name, item.stem):
        error = networks.failed_networks.get(key)
        if error:
            return f"failed to load: {error}"
    return ""


def save_generated(item: plan.Item, image, infotext: str) -> list[str]:
    """Write <name>.preview.png with the infotext; older previews that would hide it are set aside."""
    moved = plan.set_aside(plan.displaced_previews(item))
    base, ext = os.path.splitext(item.target)
    temp = f"{base}.tmp{ext}"
    images.save_image_with_geninfo(image, infotext, temp)
    os.replace(temp, item.target)
    return moved


def preview_processing(p, item: plan.Item, seed: int, subseed: int, save_outputs: bool):
    """A copy of the txt2img job for one preview: same settings, own prompt, one image, no hires fix."""
    copy_p = copy.copy(p)
    copy_p.prompt = item.full_prompt
    copy_p.negative_prompt = item.prompt.negative
    copy_p.styles = []
    copy_p.enable_hr = False
    copy_p.batch_size = 1
    copy_p.n_iter = 1
    copy_p.seed = seed
    copy_p.subseed = subseed
    copy_p.do_not_save_grid = True
    copy_p.do_not_save_samples = not save_outputs
    # process_images pops entries from these (sd_vae), and scripts add to them: one per pass.
    copy_p.override_settings = dict(p.override_settings or {})
    copy_p.extra_generation_params = dict(p.extra_generation_params or {})
    return copy_p


def generate(p, pending: list[plan.Item], result: report.RunResult, gallery: list, infotexts: list) -> None:
    if not pending:
        return
    processing.fix_seed(p)  # one seed for every preview, so they compare
    seed, subseed = p.seed, p.subseed
    save_outputs = bool(getattr(shared.opts, OPTION_SAVE_OUTPUTS, False))

    with LoadRecorder() as recorder:
        for number, item in enumerate(pending, 1):
            if state.interrupted or state.stopping_generation:
                result.not_reached = len(pending) - number + 1
                result.stopped = "interrupted" if state.interrupted else "stopped after the current image"
                return
            state.job = f"LoRA {number}/{len(pending)}: {item.stem}"
            log(f"{number}/{len(pending)} {item.full_prompt}")

            # Make Forge load the LoRA again even if the same one was applied last time, so the recorder sees it.
            if getattr(shared, "sd_model", None) is not None and hasattr(shared.sd_model, "current_lora_hash"):
                shared.sd_model.current_lora_hash = str([])
            recorder.expect(item.path)
            try:
                proc = processing.process_images(preview_processing(p, item, seed, subseed, save_outputs))
            except Exception as e:  # noqa: BLE001 - no model, out of memory...: report it and stop, keep the summary
                import traceback

                if recorder.result(item.path) is False:
                    # The LoRA did not fit and sampling was cut short; whatever broke after that is not worth stopping for.
                    state.skipped = False
                    reason = "did not fit the loaded model (Forge reported a LoRA mismatch)"
                    result.not_applied.append((item, reason))
                    log(f"not applied: {item.path}: {reason}; the aborted pass ended in {type(e).__name__}: {e}")
                    continue
                log(f"generation failed for {item.path}:\n" + traceback.format_exc())
                result.failed.append((item, f"{type(e).__name__}: {e}"))
                result.not_reached = len(pending) - number
                result.stopped = "stopped by an error (traceback in the console)"
                return

            if state.interrupted:
                result.not_reached = len(pending) - number + 1
                result.stopped = "interrupted"
                return
            applied = recorder.result(item.path)
            if not applied:
                state.skipped = False
                reason = failed_load_reason(item) or ("did not fit the loaded model (Forge reported a LoRA mismatch)" if applied is False else "Forge did not load it")
                result.not_applied.append((item, reason))
                log(f"not applied: {item.path}: {reason}")
                continue
            if state.skipped:
                state.skipped = False
                result.skipped.append(item)
                continue

            index = getattr(proc, "index_of_first_image", 0) or 0
            if len(proc.images) <= index:
                result.failed.append((item, "no image came back"))
                continue
            image = proc.images[index]
            infotext = proc.infotexts[index] if index < len(proc.infotexts) else ""
            try:
                for old in save_generated(item, image, infotext):
                    log(f"set aside {old}")
            except OSError as e:
                result.failed.append((item, str(e)))
                continue
            result.generated.append(item)
            gallery.append(image)
            infotexts.append(infotext)

            if state.stopping_generation:
                result.not_reached = len(pending) - number
                result.stopped = "stopped after the current image"
                return


# ---------------------------------------------------------------- script


class LoraPreviewsScript(scripts.Script):
    def title(self):
        return TITLE

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        with gr.Row(elem_id=self.elem_id("folder_row")):
            folder = gr.Dropdown(label="LoRA folder (worked recursively)", choices=refresh_folders(), value=None, elem_id=self.elem_id("folder"))
            refresh = ToolButton(value=refresh_symbol, elem_id=self.elem_id("refresh"), tooltip="Refresh the folder list")
        with gr.Row():
            overwrite = gr.Checkbox(value=False, label="Overwrite existing previews", elem_id=self.elem_id("overwrite"))
            use_thumbnails = gr.Checkbox(value=True, label="Use embedded thumbnails", elem_id=self.elem_id("use_thumbnails"))
            max_count = gr.Number(value=0, precision=0, minimum=0, label="Maximum LoRAs to generate (0 = all)", elem_id=self.elem_id("max_count"))
        dry = gr.Button("Dry run: list what would be done", elem_id=self.elem_id("dry_run"))
        output = gr.HTML(elem_id=self.elem_id("dry_run_output"))

        def on_refresh(current):
            choices = refresh_folders()
            return gr.update(choices=choices, value=current if current in choices else None)

        refresh.click(fn=on_refresh, inputs=[folder], outputs=[folder], show_progress=False)
        dry.click(fn=dry_run, inputs=[folder, overwrite, use_thumbnails, max_count], outputs=[output])
        return [folder, overwrite, use_thumbnails, max_count]

    def run(self, p, label, overwrite, use_thumbnails, max_count):
        folder = resolve_folder(label)
        if folder is None:
            return Processed(p, [], p.seed, info="LoRA previews: choose a LoRA folder in the script panel.")

        items = make_plan(folder, overwrite, use_thumbnails)
        to_extract = [i for i in items if i.action == plan.EXTRACT]
        pending, later = report.limited(items, _max_count(max_count))
        result = report.RunResult(later=len(later))

        # Each process_images call advances job_no once; extraction does it by hand.
        state.job_count = max(1, len(to_extract) + len(pending))
        for number, item in enumerate(to_extract, 1):
            if state.interrupted or state.stopping_generation:
                break
            state.job = f"Thumbnail {number}/{len(to_extract)}: {item.stem}"
            try:
                moved = plan.write_preview(item, item.thumbnail[0])
            except OSError as e:
                result.failed.append((item, str(e)))
                continue
            result.extracted.append(item)
            for old in moved:
                log(f"set aside {old}")
            state.nextjob()

        gallery, infotexts = [], []
        if state.interrupted or state.stopping_generation:
            result.not_reached = len(pending)
            result.stopped = "interrupted" if state.interrupted else "stopped after the current image"
        else:
            generate(p, pending, result, gallery, infotexts)

        text = report.summary(folder, items, result)
        log(text)
        prompts = [item.full_prompt for item in result.generated] or None
        return Processed(p, gallery, p.seed, info=text, all_prompts=prompts, all_seeds=[p.seed] * len(gallery) or None, infotexts=infotexts or None)
