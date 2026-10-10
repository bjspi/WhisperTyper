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


def recording_prompt_entries(entries: Any) -> List[Dict[str, Any]]:
    """Return valid prompts enabled for the recording palette, in display order.

    Each item carries ``auto_apply`` so the palette can preselect the automatic prompt.
    """
    result: List[Dict[str, Any]] = []
    for entry in load_transformations(entries):
        caption, prompt_text = entry["caption"].strip(), entry["text"].strip()
        if entry["show_during_recording"] and caption and prompt_text:
            result.append({"caption": caption, "text": prompt_text, "auto_apply": entry["auto_apply"]})
    return result


def auto_apply_prompt(entries: Any) -> Optional[str]:
    """Text of the prompt applied automatically to every transcription, if one is set."""
    return next((entry["text"] for entry in recording_prompt_entries(entries) if entry["auto_apply"]), None)


#: Maximum number of transformation templates (hotkey palette / recording palette).
MAX_TRANSFORMATIONS = 10


def transformation_entry(entry: Any) -> Dict[str, Any]:
    """Canonical stored form of one transformation template."""
    source = entry if isinstance(entry, dict) else {}
    return {
        "caption": str(source.get("caption", "")),
        "text": str(source.get("text", "")),
        "show_during_recording": source.get("show_during_recording") is True,
        "auto_apply": source.get("auto_apply") is True,
    }


def load_transformations(entries: Any) -> List[Dict[str, Any]]:
    """Return the stored templates in canonical form, capped to ``MAX_TRANSFORMATIONS``.

    At most one template applies automatically (the first one marked); it is always
    shown during recording, so it can be deselected there.
    """
    if not isinstance(entries, list):
        return []
    result = [transformation_entry(entry) for entry in entries if isinstance(entry, dict)][:MAX_TRANSFORMATIONS]
    auto_seen = False
    for entry in result:
        if entry["auto_apply"]:
            entry["auto_apply"] = not auto_seen
            entry["show_during_recording"] = entry["show_during_recording"] or not auto_seen
            auto_seen = True
    return result


def captioned_transformations(entries: Any) -> List[Dict[str, Any]]:
    """Return the templates that can appear as buttons (they need a caption)."""
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict) and str(entry.get("caption", "")).strip()]
