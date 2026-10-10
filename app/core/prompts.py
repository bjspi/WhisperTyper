"""Default prompts per UI language + swap helpers. Pure data/logic."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# --- Default Prompts ---
# Each default prompt is provided per UI language. When the user switches the UI language
# and has NOT manually edited a prompt (i.e. it still matches one of these known defaults),
# the prompt is swapped to the new language's default automatically.
DEFAULT_TRANSCRIPTION_PROMPTS = {
    "en": """The following is a transcription of a voice input. The transcription should be almost perfect to the original, only filler words and silence/emptiness should be removed. Please pay attention to spelling, capitalization, and sensible punctuation, including periods and commas. I also use "Germanized" English terms, especially from the tech and IT scene, from the areas of gadgets, smartphones, automotive, AI, and Python programming. Please recognize these as well.""",
    "de": """Das Folgende ist eine Transkription einer Spracheingabe. Die Transkription sollte nahezu perfekt dem Original entsprechen, nur Füllwörter und Stille/Leere sollten entfernt werden. Bitte achte auf Rechtschreibung, Groß- und Kleinschreibung sowie sinnvolle Zeichensetzung einschließlich Punkten und Kommas. Ich verwende außerdem „eingedeutschte“ englische Begriffe, besonders aus der Tech- und IT-Szene, aus den Bereichen Gadgets, Smartphones, Automotive, KI und Python-Programmierung. Bitte erkenne auch diese.""",
    "es": """Lo siguiente es una transcripción de una entrada de voz. La transcripción debe ser casi perfecta respecto al original; solo se deben eliminar las muletillas y los silencios/vacíos. Presta atención a la ortografía, el uso de mayúsculas y una puntuación sensata, incluidos puntos y comas. También utilizo términos ingleses adaptados, especialmente del ámbito tecnológico y de TI, de las áreas de gadgets, smartphones, automoción, IA y programación en Python. Reconócelos también.""",
    "fr": """Ce qui suit est une transcription d'une entrée vocale. La transcription doit être presque parfaitement fidèle à l'original ; seuls les mots de remplissage et les silences/vides doivent être supprimés. Veille à l'orthographe, aux majuscules et à une ponctuation sensée, y compris les points et les virgules. J'utilise aussi des termes anglais adaptés, notamment issus de la scène tech et IT, des domaines des gadgets, smartphones, automobile, IA et programmation Python. Merci de les reconnaître également.""",
}

DEFAULT_LIVEPROMPT_SYSTEM_PROMPTS = {
    "en": """You are a helpful assistant. The user will provide a direct instruction as prompt and execute it. Generate only the response to the instruction.""",
    "de": """Du bist ein hilfreicher Assistent. Der Nutzer gibt eine direkte Anweisung als Prompt und führt sie aus. Generiere nur die Antwort auf die Anweisung.""",
    "es": """Eres un asistente útil. El usuario proporcionará una instrucción directa como prompt y la ejecutará. Genera únicamente la respuesta a la instrucción.""",
    "fr": """Tu es un assistant utile. L'utilisateur fournira une instruction directe comme prompt et l'exécutera. Génère uniquement la réponse à l'instruction.""",
}

# English defaults remain available under the original names for backwards compatibility.
DEFAULT_TRANSCRIPTION_PROMPT = DEFAULT_TRANSCRIPTION_PROMPTS["en"]
DEFAULT_LIVEPROMPT_SYSTEM_PROMPT = DEFAULT_LIVEPROMPT_SYSTEM_PROMPTS["en"]


def _default_prompt_for(prompt_map: Dict[str, str], lang_code: str) -> str:
    """Return the default prompt for a language, falling back to English."""
    return prompt_map.get(lang_code, prompt_map["en"]).strip()


def _is_known_default_prompt(prompt_map: Dict[str, str], text: str) -> bool:
    """Return True if the given text matches one of the known default prompts (any language)."""
    normalized = (text or "").strip()
    return any(normalized == value.strip() for value in prompt_map.values())


#: Entry kinds: a template that rewrites the text, and the instruction (LivePrompt) entry that
#: carries the text out as an order.
PROMPT = "prompt"
INSTRUCTION = "instruction"

DEFAULT_INSTRUCTION_CAPTIONS = {
    "en": "✨ Instruction",
    "de": "✨ Anweisung",
    "es": "✨ Instrucción",
    "fr": "✨ Instruction",
}
DEFAULT_TRIGGER_WORDS = "prompt, "
DEFAULT_TRIGGER_SCAN_DEPTH = 5
_SCAN_DEPTH_RANGE = (1, 99)

#: Maximum number of own templates; the instruction entry comes on top.
MAX_TRANSFORMATIONS = 10


def default_instruction_entry(lang_code: str) -> Dict[str, Any]:
    """The instruction entry a fresh configuration starts with, in the UI language."""
    return transformation_entry({
        "kind": INSTRUCTION,
        "caption": _default_prompt_for(DEFAULT_INSTRUCTION_CAPTIONS, lang_code),
        "text": _default_prompt_for(DEFAULT_LIVEPROMPT_SYSTEM_PROMPTS, lang_code),
        "enabled": True,
    })


def _scan_depth(value: Any) -> int:
    """A valid trigger scan depth (number of leading words searched)."""
    low, high = _SCAN_DEPTH_RANGE
    if isinstance(value, int) and not isinstance(value, bool) and low <= value <= high:
        return value
    return DEFAULT_TRIGGER_SCAN_DEPTH


def transformation_entry(entry: Any) -> Dict[str, Any]:
    """Canonical stored form of one template or of the instruction entry."""
    source = entry if isinstance(entry, dict) else {}
    canonical: Dict[str, Any] = {
        "kind": INSTRUCTION if source.get("kind") == INSTRUCTION else PROMPT,
        "caption": str(source.get("caption", "")),
        "text": str(source.get("text", "")),
        "show_during_recording": source.get("show_during_recording") is True,
        "auto_apply": source.get("auto_apply") is True,
    }
    if canonical["kind"] == INSTRUCTION:
        canonical.update(
            auto_apply=False,  # the instruction runs on a trigger word or an explicit choice only
            enabled=source.get("enabled", True) is True,
            trigger_words=str(source.get("trigger_words", DEFAULT_TRIGGER_WORDS)),
            scan_depth=_scan_depth(source.get("scan_depth")),
            strip_trigger=source.get("strip_trigger") is True,
            use_selection_context=source.get("use_selection_context") is True,
        )
    return canonical


def load_transformations(entries: Any) -> List[Dict[str, Any]]:
    """Return the stored entries in canonical form, keeping their order.

    There is exactly one instruction entry (added with English defaults if missing; the
    configuration migration adds it in the UI language). At most ``MAX_TRANSFORMATIONS`` own
    templates are kept, and at most one applies automatically (the first one marked); it is
    always shown during recording, so it can be deselected there.
    """
    result: List[Dict[str, Any]] = []
    own_templates = 0
    instruction_seen = auto_seen = False
    for raw in entries if isinstance(entries, list) else []:
        if not isinstance(raw, dict):
            continue
        entry = transformation_entry(raw)
        if entry["kind"] == INSTRUCTION:
            if instruction_seen:
                continue
            instruction_seen = True
        else:
            if own_templates >= MAX_TRANSFORMATIONS:
                continue
            own_templates += 1
            if entry["auto_apply"]:
                entry["auto_apply"] = not auto_seen
                entry["show_during_recording"] = entry["show_during_recording"] or not auto_seen
                auto_seen = True
        result.append(entry)
    if not instruction_seen:
        result.insert(0, default_instruction_entry("en"))
    return result


def instruction_entry(entries: Any) -> Dict[str, Any]:
    """The instruction (LivePrompt) entry with its trigger settings."""
    return next(entry for entry in load_transformations(entries) if entry["kind"] == INSTRUCTION)


def _offered(entry: Dict[str, Any]) -> bool:
    """Whether an entry can appear as a button: it needs a caption and text, and an active instruction."""
    return bool(entry["caption"].strip() and entry["text"].strip()
                and (entry["kind"] == PROMPT or entry["enabled"]))


def recording_prompt_entries(entries: Any) -> List[Dict[str, Any]]:
    """Return the entries offered in the recording palette, in display order.

    Each item carries ``auto_apply`` (the palette preselects the automatic prompt) and ``kind``
    (an instruction runs the dictation as an order).
    """
    return [{"caption": entry["caption"].strip(), "text": entry["text"].strip(),
             "auto_apply": entry["auto_apply"], "kind": entry["kind"]}
            for entry in load_transformations(entries) if entry["show_during_recording"] and _offered(entry)]


def auto_apply_prompt(entries: Any) -> Optional[str]:
    """Text of the prompt applied automatically to every transcription, if one is set."""
    return next((entry["text"] for entry in recording_prompt_entries(entries) if entry["auto_apply"]), None)


def captioned_transformations(entries: Any) -> List[Dict[str, Any]]:
    """Return the entries offered as buttons in the rephrase hotkey window."""
    return [entry for entry in load_transformations(entries) if _offered(entry)]
