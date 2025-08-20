# -*- mode: python ; coding: utf-8 -*-

# NOTE: To build without the confirmation prompt, run PyInstaller with the --noconfirm flag:
# pyinstaller --noconfirm deploy.spec

# This is a custom .spec file designed to create the smallest possible executable.
# It excludes many PyQt5 and other modules that are not needed by this application.
import os
import glob
import sys

is_MACOS = sys.platform == 'darwin'

# --- Configuration Switch ---
# Set to True to build a one-folder bundle (a directory).
# Set to False to build a one-file executable.
DEPLOY_AS_DIRECTORY = True

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

if is_MACOS:
    icon_filename = 'app_icon.icns'
else:
    icon_filename = 'app_icon.ico'

icon_path = find_icon_file(project_root, icon_filename)

if not icon_path:
    print(f"WARNING: Icon file '{icon_filename}' not found anywhere in the project directory '{project_root}'. The application will not have an icon.")
    APP_ICON = None
else:
    APP_ICON = icon_path
    print(f"DEBUG: Icon found. Using icon file at: {APP_ICON}")

block_cipher = None

# List of modules to exclude to reduce size.
# We are being very aggressive here. If the app breaks, some of these
# might need to be removed from the excludes list.
pyqt_excludes = [
    'PyQt5', # Exclude PyQt5 to resolve conflict with PyQt6
]

# Exclude unused PyQt6 components to reduce size.
pyqt6_excludes = [
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
]

other_excludes = [
    'cv2',      # Exclude OpenCV, likely pulled in by pyautogui but not used.
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
    'pygame',  # Explicitly excluded as it was removed from the script
    'tkinter', # Not used
    'unittest', # Not needed for deployment
]

def get_app_data_files():
    """
    Recursively find all files in ../app and prepare them for PyInstaller's 'datas'.
    """
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
    excludes=pyqt_excludes + pyqt6_excludes + other_excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Exclude the software OpenGL implementation DLL to save space.
# This is a fallback renderer and not needed on most modern systems.
# The 'a.binaries' is a list of tuples (dest_path, src_path, type). We check the src_path.
a.binaries[:] = [x for x in a.binaries if 'opengl32sw.dll' not in os.path.basename(x[1])]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

if is_MACOS:
    # Create a macOS .app bundle
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='WhisterTyper',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=APP_ICON,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='WhisterTyper',
    )
    app = BUNDLE(
        coll,
        name='WhisterTyper.app',
        icon=APP_ICON,
        bundle_identifier='gh.bjspi.whistertyper',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'CFBundleShortVersionString': '1.0.0',
            'NSMicrophoneUsageDescription': 'This app requires microphone access to record audio for transcription.',
            'NSAccessibilityUsageDescription': 'This app needs permission for global hotkeys and clipboard management (e.g., to start transcription and copy results).',
            'NSInputMonitoringUsageDescription': 'This app requires permission to monitor keyboard input to detect global hotkeys for starting and stopping transcription.',
        },
    )
elif DEPLOY_AS_DIRECTORY:
    # This creates a one-folder bundle for Windows/Linux
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='WhisterTyper',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,  # Set to False for a GUI application
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=APP_ICON
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='WhisterTyper',
    )
else:
    # This creates a one-file executable for Windows/Linux
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        name='WhisterTyper',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        runtime_tmpdir=None,
        console=False,  # Set to False for a GUI application
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=APP_ICON
    )
