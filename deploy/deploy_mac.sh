#!/bin/bash

# This script automates the PyInstaller build process on macOS.
# It navigates to the script's directory and runs PyInstaller with the spec file.
#
# Before running, make this script executable from your terminal with:
# chmod +x deploy_mac.sh
#
# Then, run the script with:
# ./deploy_mac.sh

# Get the directory where the script is located to ensure paths are correct
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to the script's directory
cd "$DIR"

echo "Starting PyInstaller build for macOS..."

# Run PyInstaller with the spec file.
# The --noconfirm flag automatically removes the old build directory without prompting.
pyinstaller --noconfirm pyinstaller.spec

echo "Build process finished."

