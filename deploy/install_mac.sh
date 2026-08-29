#!/bin/bash
set -euo pipefail

# Build, replace and launch the signed macOS App Bundle while keeping the previous
# installation recoverable. User settings under ~/.WhisperTyper are never touched.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
TARGET="/Applications/WhisperTyper.app"
LEGACY_TARGET="/Applications/WhisterTyper.app"
BACKUP_DIR="$HOME/.WhisperTyper/app-backups"

# Preserve the installed app's identity during an update. This keeps macOS TCC grants attached
# even when an older build used a different bundle identifier or a local signing certificate.
# Fresh installs retain the project's normal defaults from pyinstaller.spec/deploy_mac.sh.
existing_app=""
for candidate in "$TARGET" "$LEGACY_TARGET"; do
    if [ -d "$candidate" ]; then
        existing_app="$candidate"
        break
    fi
done

if [ -n "$existing_app" ] && [ -z "${WHISPERTYPER_BUNDLE_IDENTIFIER:-}" ]; then
    installed_bundle_id="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' \
        "$existing_app/Contents/Info.plist" 2>/dev/null || true)"
    if [ -n "$installed_bundle_id" ]; then
        export WHISPERTYPER_BUNDLE_IDENTIFIER="$installed_bundle_id"
        echo "Preserving installed bundle identifier: $installed_bundle_id"
    fi
fi

if [ -n "$existing_app" ] && [ -z "${WHISPERTYPER_CODESIGN_IDENTITY:-}" ]; then
    installed_authority="$(codesign -dv --verbose=4 "$existing_app" 2>&1 \
        | sed -n 's/^Authority=//p' | head -1)"
    if [ -n "$installed_authority" ] \
        && security find-identity -v -p codesigning 2>/dev/null \
            | grep -Fq "\"$installed_authority\""; then
        export WHISPERTYPER_CODESIGN_IDENTITY="$installed_authority"
        echo "Preserving installed signing identity: $installed_authority"
    fi
fi

"$DIR/deploy_mac.sh"

BUILT_APP="$DIR/dist/WhisperTyper.app"
if [ ! -d "$BUILT_APP" ]; then
    echo "ERROR: Built app bundle not found: $BUILT_APP" >&2
    exit 1
fi
codesign --verify --deep --strict "$BUILT_APP"

mkdir -p "$BACKUP_DIR"
timestamp="$(date +%Y%m%d-%H%M%S)"

# Stop either spelling of the installed app. The anchored executable paths avoid
# terminating source-based Python runs or unrelated Whisper applications.
for executable in \
    "/Applications/WhisperTyper.app/Contents/MacOS/WhisperTyper" \
    "/Applications/WhisterTyper.app/Contents/MacOS/WhisterTyper"; do
    pid="$(pgrep -f "^${executable}$" || true)"
    if [ -n "$pid" ]; then
        kill -TERM "$pid"
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.2
        done
    fi
done

if [ -d "$TARGET" ]; then
    installed_version="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
        "$TARGET/Contents/Info.plist" 2>/dev/null || echo unknown)"
    mv "$TARGET" "$BACKUP_DIR/WhisperTyper-${installed_version}-${timestamp}.app"
fi
if [ -d "$LEGACY_TARGET" ]; then
    mv "$LEGACY_TARGET" "$BACKUP_DIR/WhisterTyper-legacy-${timestamp}.app"
fi

ditto "$BUILT_APP" "$TARGET"
codesign --verify --deep --strict "$TARGET"

# Keep startup behavior stable while correcting the historic visible app spelling.
osascript <<'APPLESCRIPT'
tell application "System Events"
    if exists login item "WhisterTyper" then delete login item "WhisterTyper"
    if exists login item "WhisperTyper" then delete login item "WhisperTyper"
    make login item at end with properties {path:"/Applications/WhisperTyper.app", hidden:false}
    return "Login item updated"
end tell
APPLESCRIPT

open "$TARGET"
echo "Installed and launched: $TARGET"
echo "Previous app bundles, if any, are in: $BACKUP_DIR"
