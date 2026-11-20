"""Choose the preview prompt of a LoRA: trigger, sample prompt, or cleaned file name."""

import json
import os
import re
from dataclasses import dataclass, field

from . import scan

DEFAULT_STOPWORDS = (
    "lora lokr locon loha lycoris lyco dora peft adapter "
    "rank dim alpha factor bf16 fp16 fp32 fp8 nf4 int8 int4 "
    "comfy comfyui diffusers kohya safetensors ckpt checkpoint "
    "epoch epochs step steps final last model base "
    "sd sd1 sd15 sd2 sd21 sd3 sd35 sdxl xl pony illustrious noobai "
    "flux flux1 flux2 dev schnell klein krea krea2 turbo qwen qwenimage "
    "wan wan21 wan22 moss zimage zit lumina lumina2 anima ernie hidream chroma fibo fibo2 h3"
)


@dataclass
class Templates:
    trigger: str = "An image of {trigger}"
    sample: str = "{sample}"
    name: str = "An image of {name}"
    negative: str = ""
    stopwords: str = DEFAULT_STOPWORDS
    default_weight: float = 1.0


@dataclass
class Prompt:
    source: str  # where the text came from, for the dry run
    text: str  # the filled template, without the LoRA tag
    negative: str
    weight: float
    values: dict = field(default_factory=dict)


# ---------------------------------------------------------------- sidecars


def _read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def user_metadata(lora_path: str) -> dict:
    """Forge's <lora>.json (extra_networks.get_user_metadata)."""
    data = _read_json(scan.base_path(lora_path) + ".json")
    return data if isinstance(data, dict) else {}


def civitai_trained_words(lora_path: str) -> str:
    """trainedWords from Civitai Helper's <lora>.civitai.info."""
    data = _read_json(scan.base_path(lora_path) + ".civitai.info")
    if not isinstance(data, dict):
        return ""
    words = data.get("trainedWords")
    if isinstance(words, str):
        words = [words]
    if not isinstance(words, list):
        return ""
    return ", ".join(w.strip() for w in words if isinstance(w, str) and w.strip())


# ---------------------------------------------------------------- header

# SAI model spec first, then plain keys: ai-toolkit copies its config's
# `meta:` block into the header unchanged, so a `trigger_word` there lands
# as a key of that name.
TRIGGER_KEYS = ("modelspec.trigger_phrase", "trigger_word", "trigger_words", "trigger_phrase", "trigger")


def _as_text(value) -> str:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                value = json.loads(stripped)
            except ValueError:
                return stripped
        else:
            return stripped
    if isinstance(value, list):
        return ", ".join(str(v).strip() for v in value if isinstance(v, str) and v.strip())
    return ""


def header_trigger(metadata: dict) -> tuple[str, str]:
    """(key, trigger) from the first trigger key with a value, or ("", "")."""
    for key in TRIGGER_KEYS:
        text = _as_text(metadata.get(key))
        if text:
            return key, text
    return "", ""


def _tag_frequency_tables(metadata: dict) -> list[dict]:
    tables = []
    raw = metadata.get("ss_tag_frequency")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                tables.extend(v for v in parsed.values() if isinstance(v, dict))
        except (TypeError, ValueError):
            pass
    raw = metadata.get("ss_datasets")
    if raw:
        try:
            datasets = json.loads(raw)
        except (TypeError, ValueError):
            datasets = None
        for dataset in datasets if isinstance(datasets, list) else ():
            frequency = dataset.get("tag_frequency") if isinstance(dataset, dict) else None
            if isinstance(frequency, dict):
                tables.extend(v for v in frequency.values() if isinstance(v, dict))
    return tables


MAX_TAG_LENGTH = 60


def top_tag(metadata: dict) -> str:
    """The most frequent caption tag of a Kohya LoRA, or "".

    Kohya counts tags in first-seen order, so on a tie the earliest tag wins:
    the first tag of the first caption, where a trigger usually sits. Natural
    language captions count each sentence once; a top count below 2 or a long
    "tag" means there is no tag-style trigger.
    """
    counts: dict[str, int] = {}
    for table in _tag_frequency_tables(metadata):
        for tag, count in table.items():
            tag = tag.strip()
            if tag and isinstance(count, (int, float)):
                counts[tag] = counts.get(tag, 0) + int(count)
    if not counts:
        return ""
    tag, count = max(counts.items(), key=lambda item: item[1])  # max keeps the first of equals
    if count < 2 or len(tag) > MAX_TAG_LENGTH:
        return ""
    return tag


MAX_SAMPLE_LENGTH = 500
_PROSE_START = re.compile(r"^\s*(this|these|a lora|an? \w+ lora|lora|trained|made|created|use|works|for |version|v\d)", re.IGNORECASE)


def sample_prompt(metadata: dict) -> str:
    """modelspec.description when it reads like a prompt, or "".

    Some trainers write a sample prompt there, others a sentence about the
    LoRA. A description with a link, a line break, more than 500 characters
    or a typical opening of prose is not used.
    """
    text = metadata.get("modelspec.description")
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if not text or len(text) > MAX_SAMPLE_LENGTH or "\n" in text:
        return ""
    if re.search(r"https?://|www\.", text, re.IGNORECASE) or _PROSE_START.match(text):
        return ""
    return text


# ---------------------------------------------------------------- name

_EPOCH_SUFFIX = re.compile(r"[\s_.\-]*\d{3,}$")
_NETWORK_SPELLINGS = re.compile(r"LoRA|LoKr|LoKR|LyCORIS|LoCon|LoHa|DoRA")
_SEPARATORS = re.compile(r"[\s_\-.+,;()\[\]{}]+")
_NOISE_TOKEN = re.compile(r"^(?:v\d+|r\d+|e\d+|ep\d+|\d+|\d+(?:k|e|ep|steps?|epochs?))$", re.IGNORECASE)
# Only letters followed by digits ("Krea2", "SDXL1"). A lower-to-upper split
# would break stylised names such as "HaDeS".
_LETTER_DIGIT = re.compile(r"(?<=[A-Za-z])(?=\d)")


def stopword_set(text: str) -> set[str]:
    return {w.casefold() for w in re.split(r"[\s,;]+", text or "") if w}


def clean_name(stem: str, stopwords: set[str]) -> str:
    """Words of a file name, without versions, numbers, formats and model names.

    loraname_0001234 -> "loraname"
    Murf_and_Boers_Adventures_v3_Krea2_-000017 -> "Murf and Boers Adventures"
    """

    def keep(token: str) -> bool:
        return bool(token) and token.casefold() not in stopwords and not _NOISE_TOKEN.match(token)

    text = stem
    while True:
        shorter = _EPOCH_SUFFIX.sub("", text)
        if shorter == text:
            break
        text = shorter
    text = _NETWORK_SPELLINGS.sub(" ", text)
    tokens = [t for t in _SEPARATORS.split(text) if keep(t)]
    words = []
    for token in tokens:
        words.extend(p for p in _LETTER_DIGIT.split(token) if keep(p))
    words = [w for w in words if len(w) > 1 or w.isalpha() and w.lower() in ("a", "i")]
    result = " ".join(words).strip()
    if result:
        return result
    # Nothing left: the stem with digits and separators removed.
    fallback = " ".join(t for t in _SEPARATORS.split(re.sub(r"\d+", " ", stem)) if t)
    return fallback or stem


# ---------------------------------------------------------------- choice


def _fill(template: str, values: dict) -> str:
    # Plain replace, not str.format: prompts may hold braces of their own.
    text = template
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return re.sub(r"[ \t]+", " ", text).strip()


def _weight(value, default: float) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return default
    return weight if weight else default  # Forge treats 0 as "not set"


def choose(lora_path: str, metadata: dict, templates: Templates) -> Prompt:
    user = user_metadata(lora_path)
    stem = os.path.splitext(os.path.basename(lora_path))[0]
    name = clean_name(stem, stopword_set(templates.stopwords))

    trigger, source = "", ""
    for label, value in (
        ("activation text", lambda: str(user.get("activation text") or "").strip()),
        ("civitai trainedWords", lambda: civitai_trained_words(lora_path)),
        ("header", lambda: header_trigger(metadata)),
        ("tag frequency", lambda: top_tag(metadata)),
    ):
        trigger = value()
        if isinstance(trigger, tuple):  # header_trigger names the key it used
            label, trigger = trigger
        if trigger:
            source = label
            break
    sample = "" if trigger else sample_prompt(metadata)

    values = {"trigger": trigger, "sample": sample, "name": name}
    if trigger:
        text = _fill(templates.trigger, values)
    elif sample:
        source = "modelspec.description"
        text = _fill(templates.sample, values)
    else:
        source = "name"
        text = _fill(templates.name, values)

    negative = str(user.get("negative text") or "").strip() or _fill(templates.negative, values)
    weight = _weight(user.get("preferred weight"), templates.default_weight)
    return Prompt(source=source, text=text, negative=negative, weight=weight, values=values)


WILDCARD = "{lora}"


def _substitute(text: str, values: dict) -> str:
    """Plain replace of {key} placeholders; the user's own spacing and line breaks stay.

    An empty value takes one neighbouring comma with it, so "blurry, {lora}"
    with nothing for {lora} gives "blurry", not "blurry,".
    """
    for key, value in values.items():
        placeholder = re.escape("{" + key + "}")
        if not value:
            text = re.sub(r"[ \t]*,[ \t]*" + placeholder + r"|" + placeholder + r"[ \t]*,[ \t]*", "", text)
        text = re.sub(placeholder, lambda _m, v=value: v, text)
    return text.strip()


def with_main_prompt(chosen: Prompt, main_prompt: str = "", main_negative: str = "") -> Prompt:
    """Use the txt2img prompt boxes when they hold text.

    A non-empty main prompt is the prompt for every LoRA. {lora} in it becomes
    the LoRA's own prompt from the templates (chosen.text); {trigger},
    {sample} and {name} give the bare values. The negative prompt works the
    same way, with {lora} becoming the LoRA's own negative. An empty box
    leaves the templates' result as it is.
    """
    main_prompt = (main_prompt or "").strip()
    main_negative = (main_negative or "").strip()
    if not main_prompt and not main_negative:
        return chosen
    text, negative, source = chosen.text, chosen.negative, chosen.source
    if main_prompt:
        text = _substitute(main_prompt, {"lora": chosen.text, **chosen.values})
        source = f"main prompt, {WILDCARD} = {chosen.source}" if WILDCARD in main_prompt else "main prompt"
    if main_negative:
        negative = _substitute(main_negative, {"lora": chosen.negative, **chosen.values})
    return Prompt(source=source, text=text, negative=negative, weight=chosen.weight, values=chosen.values)


def format_weight(weight: float) -> str:
    return f"{weight:g}"


def lora_prompt(alias: str, weight: float, text: str) -> str:
    """The full prompt: the LoRA tag in front, as the card puts it."""
    return f"<lora:{alias}:{format_weight(weight)}> {text}".strip()
