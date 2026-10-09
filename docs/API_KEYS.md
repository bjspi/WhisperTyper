# API key profiles

The **API Keys** settings tab sits between Transformations and General. Add a row
for each key, give it a name, choose OpenAI, Groq or Custom, and paste its key.
Key cells are masked. Keys are stored in the local `config.json` without encryption,
as they were before this change.

Transcription and Rephrasing each have an independent **Key profile** dropdown.
Only profiles for the current endpoint's provider appear. Choose a profile in each
panel and save settings. Custom endpoints use Custom profiles. Connection tests
use the current unsaved selection and edits; normal requests use saved settings.

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
