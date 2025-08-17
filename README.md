<p align="center">
  <!-- GitHub Stars Badge -->
  <a href="https://github.com/bjspi/WhisperTyper" target="_blank">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/bjspi/WhisperTyper?style=flat-square" />
  </a>

  <!-- Platform Support Badges -->
<img alt="macOS" src="https://img.shields.io/badge/-macOS-black?style=flat-square&logo=apple&logoColor=white" />
<img alt="Windows" src="https://img.shields.io/badge/-Windows-blue?style=flat-square&logo=windows&logoColor=white" />
<img alt="Linux" src="https://img.shields.io/badge/-Linux-yellow?style=flat-square&logo=linux&logoColor=white" />
</p>

# AI-based Voice Transcriber & Live Prompter

A powerful, cross-platform voice-to-text application that integrates seamlessly into your workflow. 
Use your voice to type, rephrase text, or even prompt AI models on-the-fly, all controlled by a simple hotkey and running discreetly in your system tray.

This tool is designed for developers, writers, and anyone who wants to leverage the power of AI to type faster and smarter.

- 🤫 **Discreet Systray Operation**: Runs quietly in the background.
- 🎤 **Global Voice Typing**: Transcribe your voice into any application with a single hotkey.
- 🚀 **Live Prompting**: Turn your voice into AI commands on the fly.
- ⌨️ **Clipboard Safe**: Restores your clipboard after pasting, so you never lose your copied content.
- 🔧 **Custom APIs**: Supports OpenAI, Groq, and any Whisper-compatible API.
- ✍️ **Advanced Rephrasing**: Go beyond simple transcription with powerful text transformations.

![App Screenshot](https://github.com/bjspi/WhisperTyper/blob/main/screenshot.jpg)

---

## Key Features

-   **Global Voice Typing**: Transcribe your voice into any application with a single hotkey.
-   **AI-Powered**: Supports OpenAI (Whisper, GPT models), Groq and any other Whisper-API with identical API-design for fast and accurate transcription and rephrasing.
-   **LivePrompting**: A revolutionary feature! Use trigger words to turn your speech directly into a prompt for an AI, which then types out the result.
    -   *Example*: Speak `"prompt write a short poem about rain"` and the AI will type the poem for you (based on your rephrasing prompt)
-   **Context-Aware Rephrasing**: Automatically use text you've highlighted on your screen as context for your voice prompts.
-   **Clipboard Safe**: Your clipboard is sacred. The app restores its previous content after pasting, so you never lose what you had copied. This is a major advantage over other tools that hijack your clipboard.
-   **Discreet Operation**: Runs quietly in the system tray without cluttering your taskbar (e.g., using `pythonw.exe` on Windows).
-   **Full Customization**:
    -   Customizable API endpoints, keys, models, and temperature settings for both transcription and rephrasing.
    -   Fine-tune transcription with custom prompts to improve accuracy for specific jargon or formatting.
    -   Adjustable microphone **volume gain** to boost input from quieter microphones, significantly increasing accuracy.
-   **Multi-Language UI**: The application interface is available in English, German, Spanish, and French.
-   **Cross-Platform**: Works on Windows, macOS, and Linux.

## Installation

1.  Clone the repository (or download the ZIP file):
    ```bash
    git clone https://github.com/bjspi/WhisperTyper.git
    cd VoiceTranscriber
    ```
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    > **Note for macOS users**: `PyAudio` has a dependency on `PortAudio`. If you encounter installation errors, please install it first using [Homebrew](https://brew.sh/):
    > `brew install portaudio`

    *(Note: You may need to create a `requirements.txt` file based on the project's imports.)*

## Usage

1.  Run the application. On Windows, you can use `pythonw.exe WhisterTyper.py` to run it without a console window.
2.  An icon will appear in your system tray. Right-click it to access the settings.
3.  In the **Settings** window:
    -   **Transcription Tab**: Enter your API keys, select your preferred AI model, and set your recording hotkey.
    -   **Rephrase / LivePrompt Tab**: Enable rephrasing, set your trigger words, and write a default prompt to guide the AI.
    -   **General Tab**: Choose your UI language.
4.  Save your settings. The window will hide, and the app will run in the background.
5.  Click into any text field in any application and use your hotkey to start typing with your voice!
