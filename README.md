<h2 align="center">Transform your voice into text and AI prompts</h2>
<p align="center">
    <img src="screenshots/app.png" alt="AppLogo" />
</p>

---

<p align="center">
<a href="https://github.com/bjspi/WhisperTyper/actions/workflows/ci.yml">
    <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/bjspi/WhisperTyper/ci.yml?branch=main&style=flat-square&label=CI&logo=github" />
</a>
<a href="https://github.com/bjspi/WhisperTyper/releases/latest">
    <img alt="Latest Release" src="https://img.shields.io/github/v/release/bjspi/WhisperTyper?style=flat-square&color=blue" />
</a>
<img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
<a href="LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
</a>
<a href="https://github.com/astral-sh/ruff">
    <img alt="Linted with Ruff" src="https://img.shields.io/badge/lint-ruff-261230?style=flat-square" />
</a>
<img alt="Checked with mypy" src="https://img.shields.io/badge/types-mypy-2A6DB2?style=flat-square" />
<img alt="macOS" src="https://img.shields.io/badge/-macOS-black?style=flat-square&logo=apple&logoColor=white" />
<img alt="Windows" src="https://img.shields.io/badge/-Windows-0078D4?style=flat-square&logo=windows&logoColor=white" />
<img alt="Linux" src="https://img.shields.io/badge/-Linux-FCC624?style=flat-square&logo=linux&logoColor=black" />
</p>

<p align="center">
  <a href="#--ai-based-voice-transcriber--live-prompter"><b>Introduction</b></a>
  &nbsp;·&nbsp;
  <a href="#-key-features"><b>Features</b></a>
  &nbsp;·&nbsp;
  <a href="#-screenshots"><b>Screenshots</b></a>
  &nbsp;·&nbsp;
  <a href="#%EF%B8%8F-download--installation"><b>Installation</b></a>
  &nbsp;·&nbsp;
  <a href="#%EF%B8%8F-installation-on-macos"><b>macOS</b></a>
  &nbsp;·&nbsp;
  <a href="#%EF%B8%8F-updating"><b>Updating</b></a>
  &nbsp;·&nbsp;
  <a href="#-architecture--development"><b>Development</b></a>
  &nbsp;·&nbsp;
  <a href="#--contributing"><b>Contributing</b></a>
</p>

---

# 📖  AI-based Voice Transcriber & Live Prompter
A powerful, cross-platform voice-to-text application that integrates seamlessly into your workflow. 
Use your voice to type, rephrase text, or even prompt AI models on-the-fly, all controlled by a simple hotkey and running discreetly in your system tray.

This tool is designed for developers, writers, and anyone who wants to leverage the power of AI to type faster and smarter.

- 🎤 **Voice Typing**: Transcribe your voice into any application with a single hotkey.
- 🚀 **Live Voice Prompting**: Turn your voice into AI commands on the fly (_e.g. say: "`this is a prompt; write birthday wishes for my friend`" and the AI will type it out for you_)
- ✍️ **Preset-based Rephrasing**: Go beyond transcription with custom presets for text-transformation (_select text in any editable field and use your hotkey to rephrase it_)
- 🔧 **Custom APIs**: Supports OpenAI, Groq, and many more...
- 🤫 **Systray Icon**: Runs quietly in the background without cluttering the taskbar
- ⌨️ **Clipboard Safe**: Doesn't hijack your clipboard

## 📱 Screenshots

<table>
  <tr>
    <td align="center" width="50%" valign="top">
      <a href="screenshots/app/settings-transcription.jpg" target="_blank">
        <img src="screenshots/app/settings-transcription.jpg" alt="Transcription settings: API endpoint, model, hotkey, microphone options" />
      </a>
      <br />
      <sub><b>🎤 Transcription</b> — Bring your own Whisper API (OpenAI, Groq, or any compatible endpoint), pick model &amp; temperature, set your global hotkey with optional push-to-talk, boost quiet microphones with volume gain, and keep the mic pre-warmed for instant recording starts. FFmpeg is auto-detected to enable transcribing video files.</sub>
    </td>
    <td align="center" width="50%" valign="top">
      <a href="screenshots/app/settings-rephrase-liveprompt.jpg" target="_blank">
        <img src="screenshots/app/settings-rephrase-liveprompt.jpg" alt="Rephrase and LivePrompt settings: chat API, trigger words, system prompt" />
      </a>
      <br />
      <sub><b>🚀 Rephrase / LivePrompt</b> — Configure the chat-completions API used for rephrasing, and define trigger words that turn your speech into a live AI command: start a recording with <i>“prompt, …”</i> and the AI's answer is typed right where your cursor is.</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%" valign="top">
      <a href="screenshots/app/settings-transformations.jpg" target="_blank">
        <img src="screenshots/app/settings-transformations.jpg" alt="Transformations settings: custom rephrasing presets" />
      </a>
      <br />
      <sub><b>✍️ Transformations</b> — Up to 10 custom one-click presets (translate, summarize, polish an e-mail, …). Select text in any application, press the post-rephrase hotkey, and pick the transformation from a popup menu.</sub>
    </td>
    <td align="center" width="50%" valign="top">
      <a href="screenshots/app/settings-general.jpg" target="_blank">
        <img src="screenshots/app/settings-general.jpg" alt="General settings: UI language, theme, proxy, clipboard and logging options" />
      </a>
      <br />
      <sub><b>🔧 General</b> — UI language (EN/DE/ES/FR), light/dark/system theme, input device, proxy support, log retention, and the clipboard-safe mode that restores your clipboard after every paste.</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%" valign="top">
      <a href="screenshots/app/tray-menu.jpg" target="_blank">
        <img src="screenshots/app/tray-menu.jpg" alt="System tray menu with all quick actions" width="70%" />
      </a>
      <br />
      <sub><b>🤫 System Tray</b> — The whole app lives in a tray icon: copy or re-transcribe the last recording, transcribe audio/video files, jump to the log, and self-update via <i>git pull</i> — all one click away.</sub>
    </td>
    <td align="center" width="50%" valign="top">
      <a href="screenshots/app/liveprompt-help.jpg" target="_blank">
        <img src="screenshots/app/liveprompt-help.jpg" alt="Built-in help tooltip explaining the LivePrompting feature" />
      </a>
      <br />
      <sub><b>💡 Built-in help</b> — Every non-obvious option explains itself: hover the <b>?</b> markers to learn how features like LivePrompting work, without leaving the app.</sub>
    </td>
  </tr>
</table>

## 💖 Key Features

-   **Global Voice Typing**: Transcribe your voice into any application with a single hotkey.
-   **AI-Powered**: Supports OpenAI (Whisper, GPT models), Groq and any other Whisper-API with identical API-design for fast and accurate transcription and rephrasing.
-   **LivePrompting**: A revolutionary feature! Use trigger words to turn your speech directly into a prompt for an AI, which then types out the result.
    -   *Example*: Speak `"prompt, write a short poem about rain"` and the AI will type the poem for you (based on your rephrasing prompt)
-   **Context-Aware Rephrasing**: Automatically use text you've highlighted on your screen as context for your voice prompts.
-   **Transcribe Audio & Video Files**: Pick an existing audio file (or, with **FFmpeg** installed, a video file — MP4, MOV, MKV, …) from the tray menu; the audio track is extracted to a temporary 128 kbps MP3 and transcribed. Set the FFmpeg path (or leave it empty to auto-detect on your `PATH`) on the Transcription settings page.
-   **Clipboard Safe**: Your clipboard is sacred. The app restores its previous content after pasting, so you never lose what you had copied. This is a major advantage over other tools that hijack your clipboard.
-   **Discreet Operation**: Runs quietly in the system tray without cluttering your taskbar (e.g., using `pythonw.exe` on Windows).
-   **Self-Updating**: When run from a cloned repo, a background watcher flags new versions with a green dot in the tray menu — one click updates and restarts the app.
-   **Full Customization**:
    -   Customizable API endpoints, keys, models, and temperature settings for both transcription and rephrasing.
    -   Fine-tune transcription with custom prompts to improve accuracy for specific jargon or formatting.
    -   Adjustable microphone **volume gain** to boost input from quieter microphones, significantly increasing accuracy.
-   **Multi-Language UI**: The application interface is available in English, German, Spanish, and French.
-   **Cross-Platform**: Works on Windows, macOS, and Linux.

## ⬇️ Download & Installation

> **💡 Recommended: clone the repo and run straight from the sources.** On Windows (and
> Linux) there is no need to build an .exe — the app runs perfectly from the Python
> sources, and a git clone unlocks the **built-in self-update**: WhisperTyper checks the
> repository in the background, shows a green dot on the tray menu's *Update* entry when
> a new version is available, and updates itself with one click. Alternatively, grab the
> ready-to-run ZIP from the [latest release](https://github.com/bjspi/WhisperTyper/releases/latest)
> — but you lose the self-update. (On macOS an App Bundle is recommended instead —
> for permissions reasons, see the [macOS instructions](docs/INSTALL_MACOS.md).)

1.  Clone the repository:
    ```bash
    git clone https://github.com/bjspi/WhisperTyper.git
    cd WhisperTyper
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    > **Note for macOS users**: See the full [macOS installation instructions](docs/INSTALL_MACOS.md).

    > **Optional — transcribe video files**: Install [FFmpeg](https://ffmpeg.org/download.html) to enable picking video files (MP4, MOV, MKV, …) for transcription. If it's on your `PATH` it is detected automatically; otherwise set its path on the Transcription settings page. On Windows: `winget install ffmpeg` (or `choco install ffmpeg`); on macOS: `brew install ffmpeg`; on Linux: `sudo apt install ffmpeg`.

3.  Run the application:
    ```bash
    python run.py
    ```
    **Tip (Windows):** start it with `pythonw` instead of `python` and no console (CMD)
    window appears at all — the app then lives purely in the system tray:
    ```bash
    pythonw run.py
    ```
    For a double-clickable launcher (e.g. as an autostart entry), drop this one-liner
    into a `start_hidden.cmd` next to `run.py`:
    ```bat
    start "WhisperTyper" /B pythonw.exe "%~dp0run.py" %*
    ```

## ⬇️🍏 Installation on macOS

macOS needs a few extra steps for microphone and hotkey permissions (ideally via an App
Bundle). The full guide lives in **[docs/INSTALL_MACOS.md](docs/INSTALL_MACOS.md)**.

## ↗️ Updating

**Running from a cloned repo (recommended):** WhisperTyper keeps itself up to date. A
background watcher periodically checks the repository and shows a green dot on the tray
menu's *Update* entry when a new version is available — one click pulls the update and
offers an immediate restart. No manual steps needed.

You can of course also update manually at any time:
```bash
git pull origin main
```

If you downloaded the ZIP file instead, grab the [latest release](https://github.com/bjspi/WhisperTyper/releases/latest) and replace your old files with the new ones.

If you run WhisperTyper as an App Bundle on macOS, see
[docs/INSTALL_MACOS.md](docs/INSTALL_MACOS.md#updating-the-app-bundle) for rebuilding the
bundle after an update.

## 🏗 Architecture & Development

The codebase is organized in strict layers — a pure, fully unit-tested core
(`app/core/`), self-contained worker services, thin platform adapters, and a Qt
composition root. The layering, threading model and design decisions are documented in
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

```bash
# Dev setup
pip install -e ".[dev]"

# The same checks CI runs on Linux/Windows/macOS:
ruff check app tests run.py                                # lint
mypy app/core app/services app/audio app/hotkeys app/ui    # strict type-check
pytest                                                     # 100+ headless tests
```

Every push is verified by [GitHub Actions](https://github.com/bjspi/WhisperTyper/actions):
lint (ruff), type-checks (mypy) and the test suite across Python 3.10–3.13 on all three
platforms.

## 🤝  Contributing
Feel free to contribute to this project! Whether it's fixing bugs, adding features, or improving documentation, 
your contributions are welcome — see **[CONTRIBUTING.md](CONTRIBUTING.md)** for the dev setup and ground rules.

Please follow the standard GitHub workflow: fork the repository, make your changes, and submit a pull request.

## 📄 License

Released under the [MIT License](LICENSE).
