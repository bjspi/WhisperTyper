# Search and replace

The **Replacements** settings tab edits corrections applied once to every usable
transcript, before the trigger words of the ✨ instruction (LivePrompt) are checked
and before any rephrasing. Microphone, file and batch transcriptions share this path.
Rephrasing output and text selected for the rephrase window are not corrected again.

Enable/disable corrections with the checkbox; save settings to activate edits.
The editor colors search terms blue, replacement text green and the optional
fixed-spelling field orange. Highlighting follows the theme and preserves plain text.
There are no initial rules. Use one rule per line:

```text
Croc, Krog, Krok ; Groq ; 1
chat gpt ; ChatGPT ; 1
beispiel ; muster
```

Comma-separated search terms share a replacement. Semicolons separate fields;
the optional last field is `1` for fixed spelling, or `0` (the default) to carry
over the matched text's casing. Without fixed spelling, `Beispiel` becomes `Muster`,
`BEISPIEL` becomes `MUSTER`, and `beispiel` becomes `muster`. Otherwise replacement spelling
is preserved. Search terms and replacement text must be nonempty.

Matches ignore case and cover whole words or phrases only: `beispiel` does not change
`Beispieltext`. Letters and digits form word boundaries; punctuation and underscores
separate words. Phrase whitespace may vary, including tabs and line breaks.
Search terms are literal, including regex punctuation. Longer matching terms
win. A repeated search term belongs to its last rule. Replacements are not fed
back into the search, preventing cascades such as `a → b → c`.

Malformed rules block Save and show their line number. Existing rules stay active.
If rules in a manually edited config are malformed at startup, corrections are
skipped until repaired; the editor retains the text. Rules compile on startup or
Save, not during transcription. No dependencies, model migrations or extra HTTP
requests are introduced.

The asynchronous `replacements_check` INFO record includes the operation ID,
enabled state, rule/term counts, transcript length, number of matches and matched
source-line numbers. It includes no transcript, search terms or replacements.
The latency summary reports `replacements_ms`.

Behavior follows Gboard Turbo-Type commits `0fba1cc`, `cbfa955` and `cbe000a`.
