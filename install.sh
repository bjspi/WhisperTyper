#!/usr/bin/env bash
# Set up WhisperTyper (macOS and Linux). Run from a clone, it sets up that folder. Run directly from
# the web, it clones the repository into ~/WhisperTyper first (and reuses that clone when run again):
#
#   curl -fsSL https://raw.githubusercontent.com/bjspi/WhisperTyper/main/install.sh | bash
#
# Creates a virtual environment in .venv (Python 3.10-3.13), installs the requirements into it and
# starts the tray app. It also adds the whispertyper command. Safe to run again.
#
# Uses uv if it is installed (uv also downloads Python 3.13 if needed), otherwise a Python on PATH.
#
#   ./install.sh              install and start
#   ./install.sh --no-start   install only
set -euo pipefail

REPOSITORY="https://github.com/bjspi/WhisperTyper.git"

check_portaudio() {
  # PyAudio is compiled during the installation and needs the PortAudio headers.
  if [ "$1" = "Darwin" ]; then
    local prefix
    prefix="$(brew --prefix portaudio 2>/dev/null || true)"
    if [ -z "$prefix" ] || [ ! -f "$prefix/include/portaudio.h" ]; then
      echo "PortAudio is required: brew install portaudio" >&2
      exit 1
    fi
    export CFLAGS="-I$prefix/include ${CFLAGS:-}"
    export LDFLAGS="-L$prefix/lib ${LDFLAGS:-}"
  elif [ ! -f /usr/include/portaudio.h ] && [ ! -f /usr/local/include/portaudio.h ]; then
    echo "PortAudio and a compiler are required:" >&2
    echo "  Debian/Ubuntu: sudo apt install portaudio19-dev python3-dev build-essential" >&2
    echo "  Fedora:        sudo dnf install portaudio-devel python3-devel gcc" >&2
    echo "  Arch:          sudo pacman -S portaudio base-devel" >&2
    exit 1
  fi
}

enter_project_folder() {
  # This script's folder in a clone or unpacked release; otherwise ~/WhisperTyper, cloned if needed.
  # Piped into bash, BASH_SOURCE names no file (it says "main" in a function), whatever the current folder holds.
  local script="${BASH_SOURCE[0]:-}"
  if [ -f "$script" ] && [ -f "$(dirname "$script")/requirements.txt" ]; then
    cd "$(dirname "$script")"
    return
  fi
  local target="$HOME/WhisperTyper"
  if [ -e "$target/.git" ]; then
    echo "==> Using the existing clone in $target"
  elif [ -e "$target" ]; then
    echo "$target exists but is not a clone of WhisperTyper. Move it away and run this again." >&2
    exit 1
  elif ! command -v git >/dev/null 2>&1; then
    echo "Git is required: https://git-scm.com/downloads" >&2
    exit 1
  else
    echo "==> Cloning WhisperTyper into $target"
    git clone --quiet "$REPOSITORY" "$target"
  fi
  cd "$target"
}

python_version() {
  "$1" -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>/dev/null || true
}

supported() {
  case "$1" in
    3.10|3.11|3.12|3.13) return 0 ;;
    *) return 1 ;;
  esac
}

create_venv_with_python() {
  local candidate
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && supported "$(python_version "$candidate")"; then
      "$candidate" -m venv .venv
      return
    fi
  done
  echo "Python 3.10-3.13 was not found. (Newer versions are not supported yet.)" >&2
  echo "Install Python 3.13, or uv, which brings its own Python:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
}

# Everything runs from here, the last line: when piped into bash, nothing starts before the whole
# script has arrived, and no child process can read the rest of the script from stdin.
main() {
  local start=1 argument
  for argument in "$@"; do
    case "$argument" in
      --no-start) start=0 ;;
      *) echo "unknown option: $argument" >&2; exit 2 ;;
    esac
  done

  local system
  system="$(uname)"
  case "$system" in
    MINGW*|MSYS*|CYGWIN*)
      echo "On Windows, run this in PowerShell: irm https://raw.githubusercontent.com/bjspi/WhisperTyper/main/install.ps1 | iex" >&2
      exit 1
      ;;
  esac
  check_portaudio "$system"
  enter_project_folder

  local use_uv=0 python="$PWD/.venv/bin/python"
  if command -v uv >/dev/null 2>&1; then
    use_uv=1
  fi

  if [ -x "$python" ]; then
    local version
    version="$(python_version "$python")"
    if ! supported "$version"; then
      echo "The existing .venv uses Python $version; WhisperTyper needs 3.10-3.13." >&2
      echo "Delete $PWD/.venv and run this again." >&2
      exit 1
    fi
    echo "==> Using the existing virtual environment (Python $version)"
  elif [ "$use_uv" = 1 ]; then
    echo "==> Creating the virtual environment (Python 3.13, with uv)"
    # --seed adds pip, which the app uses to install optional packages later.
    uv venv --seed --python 3.13 .venv
  else
    echo "==> Creating the virtual environment"
    create_venv_with_python
  fi

  echo "==> Installing the requirements"
  if [ "$use_uv" = 1 ]; then
    uv pip install --python "$python" -r requirements.txt
  else
    "$python" -m pip install --disable-pip-version-check -r requirements.txt
  fi

  echo "==> Adding the whispertyper command"
  # Not fatal: the app tries again at every start.
  "$python" run.py install-integration || echo "Could not add it now; WhisperTyper tries again when it starts."

  if [ "$start" = 1 ]; then
    echo "==> Starting WhisperTyper (look for its icon in the tray or menu bar)"
    nohup "$python" run.py >/dev/null 2>&1 </dev/null &
  fi
  echo "Done. From now on: whispertyper in a new terminal."
  if [ "$system" = "Darwin" ]; then
    echo "On macOS, permissions work best with an App Bundle: see docs/INSTALL_MACOS.md"
  fi
}

main "$@"
