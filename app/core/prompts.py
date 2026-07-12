"""Default prompts per UI language + swap helpers. Pure data/logic."""
from __future__ import annotations

from typing import Dict

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

DEFAULT_GENERIC_REPHRASE_PROMPTS = {
    "en": """Rephrase the following text to be more polite, professional, and clear. Correct any spelling or grammar mistakes. Return only the rephrased text.""",
    "de": """Formuliere den folgenden Text höflicher, professioneller und klarer. Korrigiere alle Rechtschreib- und Grammatikfehler. Gib nur den umformulierten Text zurück.""",
    "es": """Reformula el siguiente texto para que sea más educado, profesional y claro. Corrige cualquier error ortográfico o gramatical. Devuelve solo el texto reformulado.""",
    "fr": """Reformule le texte suivant pour qu'il soit plus poli, professionnel et clair. Corrige toutes les fautes d'orthographe ou de grammaire. Renvoie uniquement le texte reformulé.""",
}

# English defaults remain available under the original names for backwards compatibility.
DEFAULT_TRANSCRIPTION_PROMPT = DEFAULT_TRANSCRIPTION_PROMPTS["en"]
DEFAULT_LIVEPROMPT_SYSTEM_PROMPT = DEFAULT_LIVEPROMPT_SYSTEM_PROMPTS["en"]
DEFAULT_GENERIC_REPHRASE_PROMPT = DEFAULT_GENERIC_REPHRASE_PROMPTS["en"]


def _default_prompt_for(prompt_map: Dict[str, str], lang_code: str) -> str:
    """Return the default prompt for a language, falling back to English."""
    return prompt_map.get(lang_code, prompt_map["en"]).strip()


def _is_known_default_prompt(prompt_map: Dict[str, str], text: str) -> bool:
    """Return True if the given text matches one of the known default prompts (any language)."""
    normalized = (text or "").strip()
    return any(normalized == value.strip() for value in prompt_map.values())
