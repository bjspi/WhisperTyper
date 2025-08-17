# AI-based Voice Transcriber & Live Prompter

A powerful, cross-platform voice-to-text application that integrates seamlessly into your workflow. 
Use your voice to type, rephrase text, or even prompt AI models on-the-fly, all controlled by a simple hotkey and running discreetly in your system tray.

This tool is designed for developers, writers, and anyone who wants to leverage the power of AI to type faster and smarter.

- 🤫 **Discreet Systray Operation**: Runs quietly in the background.
- 🚀 **Live Prompting**: Turn your voice into AI commands on the fly.
- ⌨️ **Global Hotkeys**: Control everything with a single key press.
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

## How It Works

The workflow is designed to be as unobtrusive as possible:

1.  **(Optional) Select Context**: Highlight any text on your screen you want the AI to use as context.
2.  **Press Hotkey**: Press your configured hotkey to start recording.
3.  **Speak**: Say what you want to type, or state your trigger word followed by a prompt.
4.  **Release Hotkey**: Release the hotkey to stop recording.
5.  **AI Magic**:
    -   The app sends the audio to your chosen AI service (e.g., Whisper) for transcription.
    -   If rephrasing or LivePrompting is triggered, the text (along with any selected context) is sent to a second AI for processing.
6.  **Paste & Restore**: The final text is typed out at your cursor's location, and your original clipboard content is instantly restored.

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
