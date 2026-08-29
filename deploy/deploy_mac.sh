#!/bin/bash

# This script automates the PyInstaller build process on macOS.
# It navigates to the script's directory and runs PyInstaller with the spec file.
#
# Before running, make this script executable from your terminal with:
# chmod +x deploy_mac.sh
#
# Then, run the script with:
# ./deploy_mac.sh
#
# Optional for stable macOS permissions across app updates:
# export WHISPERTYPER_CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
# export WHISPERTYPER_ENTITLEMENTS_FILE="/absolute/path/to/entitlements.plist"

# Get the directory where the script is located to ensure paths are correct
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the script's directory
cd "$DIR"

echo "Starting PyInstaller build for macOS..."

if [ -n "${WHISPERTYPER_CODESIGN_IDENTITY:-}" ]; then
    echo "Using configured macOS signing identity: ${WHISPERTYPER_CODESIGN_IDENTITY}"
    if [[ "${WHISPERTYPER_CODESIGN_IDENTITY}" == Developer\ ID\ Application:* ]]; then
        echo "Developer ID identity detected. Hardened runtime signing remains enabled."
    else
        echo "Local/non-Developer ID identity detected. Using a post-build deep sign without hardened runtime."
        echo "This macOS extension keeps local updates stable without reproducing the silent-microphone issue."
    fi
else
    echo "WARNING: No WHISPERTYPER_CODESIGN_IDENTITY configured."
    echo "WARNING: The build will be ad hoc signed."
    echo "WARNING: On macOS this can cause Accessibility/Input Monitoring permissions"
    echo "WARNING: to stop working after replacing the .app with a new build."
fi

APP_BUNDLE="$DIR/dist/WhisperTyper.app"
ENTITLEMENTS_FILE="${WHISPERTYPER_ENTITLEMENTS_FILE:-$DIR/macos-entitlements.plist}"

# Prefer the project's environment so the bundle contains the same dependency versions
# that passed its tests. Falling back to PATH keeps the script usable for other setups.
if [ -x "$DIR/../venv/bin/pyinstaller" ]; then
    PYINSTALLER="$DIR/../venv/bin/pyinstaller"
elif [ -x "$DIR/../.venv/bin/pyinstaller" ]; then
    PYINSTALLER="$DIR/../.venv/bin/pyinstaller"
else
    PYINSTALLER="$(command -v pyinstaller || true)"
fi
if [ -z "$PYINSTALLER" ]; then
    echo "ERROR: PyInstaller is missing. Install it in the project environment first:" >&2
    echo "       python3 -m pip install pyinstaller" >&2
    exit 1
fi
echo "Using PyInstaller: $PYINSTALLER"

# Run PyInstaller with the spec file.
# The --noconfirm flag automatically removes the old build directory without prompting.
"$PYINSTALLER" --noconfirm pyinstaller.spec

if [ -n "${WHISPERTYPER_CODESIGN_IDENTITY:-}" ] && [ -d "$APP_BUNDLE" ]; then
    if [[ "${WHISPERTYPER_CODESIGN_IDENTITY}" == Developer\ ID\ Application:* ]]; then
        if [ -f "$ENTITLEMENTS_FILE" ]; then
            echo "Deep re-signing app bundle with entitlements for Developer ID launch behavior..."
            codesign --force --deep --sign "$WHISPERTYPER_CODESIGN_IDENTITY" --options runtime --entitlements "$ENTITLEMENTS_FILE" "$APP_BUNDLE"
        else
            echo "WARNING: Entitlements file not found at $ENTITLEMENTS_FILE"
            echo "WARNING: Continuing without post-build deep re-signing."
        fi
    else
        echo "Deep re-signing app bundle without hardened runtime for local macOS builds..."
        codesign --force --deep --sign "$WHISPERTYPER_CODESIGN_IDENTITY" "$APP_BUNDLE"
    fi
fi

if [ ! -d "$APP_BUNDLE" ]; then
    echo "ERROR: Expected app bundle was not produced: $APP_BUNDLE" >&2
    exit 1
fi

codesign --verify --deep --strict "$APP_BUNDLE"
echo "Built bundle identifier: $(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$APP_BUNDLE/Contents/Info.plist")"
echo "Built bundle version: $(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP_BUNDLE/Contents/Info.plist")"
echo "Build process finished."
