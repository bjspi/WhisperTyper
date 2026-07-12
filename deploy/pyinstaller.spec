# -*- mode: python ; coding: utf-8 -*-

# macOS App Bundle build spec (used by deploy_mac.sh).
#
# NOTE: To build without the confirmation prompt, run PyInstaller with the --noconfirm flag:
# pyinstaller --noconfirm pyinstaller.spec
#
# Windows/Linux users don't need this at all — WhisperTyper is meant to run straight
# from the sources there (python run.py / pythonw run.py), which also enables the
# built-in tray self-update. Only macOS benefits from a bundle, because the granted
# Input Monitoring / Accessibility / Microphone permissions attach to the bundle.
import os
import sys

is_MACOS = sys.platform == 'darwin'
if not is_MACOS:
    raise SystemExit(
        "This spec builds the macOS App Bundle only. On Windows/Linux run WhisperTyper "
        "directly from the sources: python run.py (see README)."
    )

CODESIGN_IDENTITY = os.environ.get("WHISPERTYPER_CODESIGN_IDENTITY", "").strip() or None
DEFAULT_ENTITLEMENTS_FILE = os.path.join(os.path.dirname(SPECPATH), 'macos-entitlements.plist')
ENTITLEMENTS_FILE = os.environ.get("WHISPERTYPER_ENTITLEMENTS_FILE", "").strip() or None
if not ENTITLEMENTS_FILE and os.path.exists(DEFAULT_ENTITLEMENTS_FILE):
    ENTITLEMENTS_FILE = DEFAULT_ENTITLEMENTS_FILE
USES_DEVELOPER_ID_RUNTIME = bool(CODESIGN_IDENTITY and CODESIGN_IDENTITY.startswith("Developer ID Application:"))
PYINSTALLER_CODESIGN_IDENTITY = CODESIGN_IDENTITY if USES_DEVELOPER_ID_RUNTIME else None
PYINSTALLER_ENTITLEMENTS_FILE = ENTITLEMENTS_FILE if USES_DEVELOPER_ID_RUNTIME else None

# --- Icon Path ---
# Search for the icon file within the project directory to make the path resolution more robust.
def find_icon_file(root_path, icon_name):
    """Search for a file in a directory and its subdirectories."""
    for root, dirs, files in os.walk(root_path):
        if icon_name in files:
            return os.path.join(root, icon_name)
    return None

# SPECPATH is a global variable from PyInstaller. Project root is one level up from the 'deploy' dir.
project_root = os.path.abspath(os.path.join(os.path.dirname(SPECPATH), '..'))
APP_ICON = find_icon_file(project_root, 'app_icon.icns')
if not APP_ICON:
    print("WARNING: Icon file 'app_icon.icns' not found in the project directory. "
          "The application will not have an icon.")
else:
    print(f"DEBUG: Icon found. Using icon file at: {APP_ICON}")

block_cipher = None

# Exclude unused modules to reduce bundle size.
excludes = [
    'PyQt5',  # resolve conflict with PyQt6
    'PyQt6.QtDesigner',
    'PyQt6.QtHelp',
    'PyQt6.QtMultimedia',
    'PyQt6.QtMultimediaWidgets',
    'PyQt6.QtOpenGL',
    'PyQt6.QtOpenGLWidgets',
    'PyQt6.QtPrintSupport',
    'PyQt6.QtSql',
    'PyQt6.QtSvg',
    'PyQt6.QtTest',
    'PyQt6.QtWebEngineCore',
    'PyQt6.QtWebEngineWidgets',
    'PyQt6.QtXml',
    'cv2',      # likely pulled in by pyautogui but not used
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
    'pygame',
    'tkinter',
    'unittest',
]

def get_app_data_files():
    """Recursively find all files in ../app and prepare them for PyInstaller's 'datas'."""
    data_files = []
    app_dir = os.path.join('..', 'app')
    for root, dirs, files in os.walk(app_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # Destination path inside the bundle should be relative to app_dir
            dest_dir = os.path.relpath(root, '..')
            data_files.append((file_path, dest_dir))
    return data_files

a = Analysis(
    [os.path.join('..', 'run.py')],
    pathex=[],
    binaries=[],
    datas=get_app_data_files(),
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if CODESIGN_IDENTITY:
    print(f"DEBUG: Using macOS code signing identity: {CODESIGN_IDENTITY}")
    if USES_DEVELOPER_ID_RUNTIME:
        print("DEBUG: Developer ID identity detected. PyInstaller runtime signing remains enabled.")
    else:
        print("DEBUG: Local/non-Developer ID identity detected. Skipping PyInstaller runtime signing")
        print("DEBUG: and relying on a final post-build deep sign to preserve microphone/TCC behavior.")
else:
    print("WARNING: No macOS code signing identity configured. The app will be ad hoc signed,")
    print("WARNING: which can cause macOS Accessibility/Input Monitoring permissions to reset")
    print("WARNING: whenever the .app bundle is replaced by a new build.")
if PYINSTALLER_ENTITLEMENTS_FILE:
    print(f"DEBUG: Using macOS entitlements file: {PYINSTALLER_ENTITLEMENTS_FILE}")

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WhisperTyper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=PYINSTALLER_CODESIGN_IDENTITY,
    entitlements_file=PYINSTALLER_ENTITLEMENTS_FILE,
    icon=APP_ICON,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='WhisperTyper',
)
app = BUNDLE(
    coll,
    name='WhisperTyper.app',
    icon=APP_ICON,
    bundle_identifier='gh.bjspi.whispertyper',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'CFBundleShortVersionString': '0.5.0',
        'NSMicrophoneUsageDescription': 'This app requires microphone access to record audio for transcription.',
        'NSAccessibilityUsageDescription': 'This app needs permission for global hotkeys and clipboard management (e.g., to start transcription and copy results).',
        'NSInputMonitoringUsageDescription': 'This app requires permission to monitor keyboard input to detect global hotkeys for starting and stopping transcription.',
        'NSAppleEventsUsageDescription': 'This app uses Apple Events to return focus to the previous app after transcription or rephrasing.',
    },
)
