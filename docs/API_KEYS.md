# API key profiles

The **API Keys** settings tab sits between Transformations and General. Add a row
for each key, give it a name, choose OpenAI, Groq or Custom, and paste its key.
Key cells and profile dropdowns show the first 10 and last 4 characters, with the
middle masked. Keys of 14 characters or fewer stay fully masked. The password
input beneath each preview allows editing without revealing the full key.
Keys are stored in the local `config.json` without encryption,
as they were before this change.

Transcription and Rephrasing each have an independent **Key profile** dropdown.
Only profiles for the current endpoint's provider appear. Choose a profile in each
panel and save settings. Custom endpoints use Custom profiles. Connection tests
use the current unsaved selection and edits; normal requests use saved settings.

Both panels have a **Provider** dropdown with OpenAI, Groq and Custom. Selecting
OpenAI or Groq sets its endpoint URL. Selecting Custom clears the URL for entry;
you can also edit URLs directly, which updates the provider dropdown.
On a provider change, the matching saved valid key profile is selected, otherwise
the first valid matching key in table order. Empty and invalid keys are skipped.
Without a valid matching key the selection stays empty. Same-provider edits keep
your current selection. Review the chosen profile/model and save. Rephrasing
settings apply to LivePrompt, general rephrasing and transformation presets.

Both model dropdowns show suggestions only for the endpoint's provider and are
directly editable: select a suggestion using the arrow or type your own model
name in the same field. Saved and custom model names are preserved. When changing
providers, an incompatible built-in model switches to the first suggestion for
the new provider. Custom endpoints have no provider model suggestions.
Model names are displayed without provider suffixes. Legacy saved names with
suffixes still resolve to the same models.

OpenAI transcription suggestions are `gpt-transcribe`, `gpt-4o-transcribe`,
`gpt-4o-mini-transcribe`, `gpt-4o-mini-transcribe-2025-12-15`,
`gpt-4o-transcribe-diarize` and `whisper-1`. The first is the documented recommended
file-transcription model. OpenAI rephrasing defaults to **`gpt-6-luna`** for new
settings and explicit provider changes. Existing models are never migrated;
choose another model yourself in the dropdown.
Groq chat suggestions are fixed constants: `openai/gpt-oss-120b`,
`openai/gpt-oss-20b`, `llama-3.3-70b-versatile` and `llama-3.1-8b-instant`.

The suggestions are a curated local list, verified on 2026-10-09, rather than an
account-specific list fetched using your key. Account access and future provider
changes can affect availability; enter another supported name when needed.
Sources: [OpenAI models](https://developers.openai.com/api/docs/models),
[OpenAI file transcription](https://developers.openai.com/api/docs/guides/speech-to-text),
[Groq models](https://console.groq.com/docs/models), and
[Groq speech to text](https://console.groq.com/docs/speech-to-text).

GPT-5.6 and GPT-6 chat models use their default reasoning/temperature settings;
the temperature control is disabled and the request omits that parameter.
`gpt-transcribe` uses `languages[]` for the selected language, preserves the
transcription prompt, and omits temperature. `gpt-4o-transcribe-diarize` omits
prompt and temperature, requests plain transcript JSON and uses server-side
`chunking_strategy=auto` for longer recordings; the UI disables unsupported
controls. The application still uploads the recording as one direct request.
Other transcription models keep their existing `language` and temperature
parameters. See the
[OpenAI GPT-6 parameter guidance](https://developers.openai.com/api/docs/guides/latest-model#gpt-6-astra-update-api-and-model-parameters).

Renaming a profile keeps its selections. Removing a selected profile clears the
corresponding dropdown, so another key is never selected implicitly. Empty keys
are allowed while entering profiles, but cannot make requests. Every profile must
have a name before settings can be saved.

Existing inline transcription and rephrasing keys are migrated on startup into
named profiles with stable IDs. Identical keys for the same provider share one
profile. Keys for different providers stay separate. The former inline credential
fields are removed from the saved config. A missing rephrasing key is not replaced
by a transcription key.

## Optional Groq rotation

Enable **Rotate Groq keys for transcription** in the API Keys tab and save. The
first request uses the selected transcription key; subsequent requests cycle
through all distinct, nonempty Groq keys in table order. Duplicate key values and
invalid header values are skipped. Rotation is disabled by default and only
applies to a Groq transcription endpoint with an explicitly selected valid profile.
OpenAI and Custom endpoints use their selected key.

Microphone/file transcription and manual retries take the next key when their
worker is constructed. In-flight workers keep their own credential snapshots.
Changing the selected key or key pool restarts the cycle at the chosen key.
Rephrasing/transformations and connection tests always use their explicitly
selected profile. Rotation does not add automatic retries or failure-based key
switching. All configured Groq keys participate when rotation is enabled.

The INFO record `transcription_credential` links each operation's `op` ID to the
chosen `profile_id`, provider and rotation flag. The profile ID is also stored
with its row in `config.json`. Key values and profile names are never included in
this log record.
