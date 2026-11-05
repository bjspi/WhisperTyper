<#
.SYNOPSIS
    Sets up WhisperTyper (Windows).

.DESCRIPTION
    Run from a clone, it sets up that folder. Run directly from the web, it clones the
    repository into ~\WhisperTyper first (and reuses that clone when run again):

        irm https://raw.githubusercontent.com/bjspi/WhisperTyper/main/install.ps1 | iex

    Creates a virtual environment in .venv (Python 3.10-3.13), installs the requirements
    into it and starts the tray app. At its first start the app adds the whispertyper
    command and the Start menu entry. Safe to run again.

    Uses uv if it is installed (uv also downloads Python 3.13 if needed), otherwise a
    Python found through the py launcher or on PATH. If there is none, it asks for the
    path of one, for example on a network share.

.PARAMETER NoStart
    Do not start WhisperTyper after installing.

.PARAMETER Python
    The Python to create the virtual environment with: python.exe or its folder.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1 -Python \\server\tools\Python313\python.exe
#>
[CmdletBinding()]
param(
    [switch]$NoStart,
    [string]$Python
)

# The script block keeps settings and helpers out of the user's session when run through iex.
& {
    param([bool]$NoStart, [string]$ScriptFolder, [string]$CustomPython)

    $ErrorActionPreference = 'Stop'
    $repository = 'https://github.com/bjspi/WhisperTyper.git'
    $supportedVersions = '3.13', '3.12', '3.11', '3.10'
    $uvInstalled = [bool](Get-Command uv -ErrorAction SilentlyContinue)


    function Get-ProjectFolder {
        # This script's folder in a clone or unpacked release; otherwise ~\WhisperTyper, cloned if needed.
        if ($ScriptFolder -and (Test-Path (Join-Path $ScriptFolder 'requirements.txt'))) { return $ScriptFolder }
        $target = Join-Path $HOME 'WhisperTyper'
        if (Test-Path (Join-Path $target '.git')) {
            Write-Host "==> Using the existing clone in $target"
            return $target
        }
        if (Test-Path $target) { throw "$target exists but is not a clone of WhisperTyper. Move it away and run this again." }
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            throw "Git is required: winget install Git.Git`n(Open a new terminal afterwards and run this again.)"
        }
        Write-Host "==> Cloning WhisperTyper into $target"
        git clone --quiet $repository $target | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Cloning the repository failed.' }
        return $target
    }


    function Get-PythonVersion {
        # "3.13" for a working interpreter; nothing for a missing one or the Microsoft Store stub.
        param([string]$Program, [string[]]$Arguments = @())
        $ErrorActionPreference = 'Continue'
        try {
            $version = & $Program @Arguments -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0) { return "$version".Trim() }
        }
        catch {
            return
        }
    }


    function Get-PythonProblem {
        # Why this Python cannot create the environment; nothing when it can.
        param([string]$Program, [string[]]$Arguments = @())
        # Windows' "App execution aliases" for the Microsoft Store are no Python to build on.
        if ($Program -like '*\Microsoft\WindowsApps\*') { return 'is the Windows Store alias, not an installed Python.' }
        $version = Get-PythonVersion $Program $Arguments
        if (-not $version) { return 'does not run.' }
        if ($supportedVersions -notcontains $version) { return "is Python $version, but WhisperTyper needs 3.10-3.13." }
        $ErrorActionPreference = 'Continue'
        # venv brings pip through ensurepip; reduced Pythons (the embeddable package, for example) lack them.
        & $Program @Arguments -c 'import venv, ensurepip' 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { return 'has no venv or ensurepip module, so it cannot create the environment.' }
    }


    function New-Venv {
        param([string]$Venv, [string]$Program, [string[]]$Arguments = @())
        & $Program @Arguments -m venv $Venv | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Creating the virtual environment failed.' }
    }


    function New-VenvWithFoundPython {
        # Create the venv with the first usable Python: py launcher (newest first), then PATH; False if none.
        param([string]$Venv)
        $candidates = @($supportedVersions | ForEach-Object { , @('py', "-$_") })
        foreach ($command in Get-Command python, python3 -All -CommandType Application -ErrorAction SilentlyContinue) {
            $candidates += , @($command.Source)
        }
        foreach ($candidate in $candidates) {
            $program = $candidate[0]
            $arguments = @($candidate | Select-Object -Skip 1)
            if (Get-PythonProblem $program $arguments) { continue }
            New-Venv $Venv $program $arguments
            return $true
        }
        return $false
    }


    function Resolve-PythonPath {
        # The python.exe that a given path means (quotes, variables and its folder are fine); explains if unusable.
        param([string]$Text)
        $path = [Environment]::ExpandEnvironmentVariables($Text.Trim().Trim('"', "'"))
        if (Test-Path -LiteralPath $path -PathType Container) { $path = Join-Path $path 'python.exe' }
        if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or (Split-Path -Leaf $path) -notlike 'python*.exe') {
            Write-Host "No python.exe found at $path" -ForegroundColor Yellow
            return
        }
        $problem = Get-PythonProblem $path
        if ($problem) {
            Write-Host "$path $problem" -ForegroundColor Yellow
            return
        }
        return $path
    }


    function Read-PythonPath {
        # Ask until the path is usable; nothing when the user leaves it empty or nobody can answer.
        Write-Host 'Python 3.10-3.13 was not found on this computer''s PATH.' -ForegroundColor Yellow
        Write-Host 'If you have one somewhere else, for example on a network share, enter its path'
        Write-Host '(python.exe or its folder). Leave it empty to cancel.'
        while ($true) {
            try {
                $answer = Read-Host 'Path to Python'
            }
            catch {
                return
            }
            if (-not $answer -or -not $answer.Trim()) { return }
            $path = Resolve-PythonPath $answer
            if ($path) { return $path }
        }
    }


    try {
        $root = Get-ProjectFolder
        $venv = Join-Path $root '.venv'
        $venvPython = Join-Path $venv 'Scripts\python.exe'
        $createdWithUv = $false

        if (Test-Path $venvPython) {
            $version = Get-PythonVersion $venvPython
            if ($supportedVersions -notcontains $version) {
                throw "The existing .venv uses Python $version; WhisperTyper needs 3.10-3.13.`nDelete $venv and run this again."
            }
            Write-Host "==> Using the existing virtual environment (Python $version)"
        }
        elseif ($CustomPython) {
            $path = Resolve-PythonPath $CustomPython
            if (-not $path) { throw 'The Python given with -Python cannot be used.' }
            Write-Host "==> Creating the virtual environment with $path"
            New-Venv $venv $path
        }
        else {
            if ($uvInstalled) {
                Write-Host '==> Creating the virtual environment (Python 3.13, with uv)'
                # --seed adds pip, which the app uses to install optional packages later.
                uv venv --seed --python 3.13 $venv
                $createdWithUv = $LASTEXITCODE -eq 0
                if (-not $createdWithUv) { Write-Host 'uv could not create it (no download allowed?). Looking for a Python instead.' -ForegroundColor Yellow }
            }
            if (-not $createdWithUv) {
                Write-Host '==> Creating the virtual environment'
                if (-not (New-VenvWithFoundPython $venv)) {
                    $path = Read-PythonPath
                    if (-not $path) {
                        throw ("No usable Python. (3.14 does not work yet: PyAudio has no ready-made package for it.)`n" +
                            "Install Python 3.13 from https://www.python.org/downloads/ or uv, which brings its own Python:`n" +
                            '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"')
                    }
                    New-Venv $venv $path
                }
            }
        }

        Write-Host '==> Installing the requirements'
        $requirements = Join-Path $root 'requirements.txt'
        # The environment's own pip respects pip.ini (a company package index, for example); uv does not.
        if ($createdWithUv -or ($uvInstalled -and -not (Test-Path (Join-Path $venv 'Scripts\pip.exe')))) {
            uv pip install --python $venvPython -r $requirements
        }
        else {
            & $venvPython -m pip install --disable-pip-version-check -r $requirements
        }
        if ($LASTEXITCODE -ne 0) { throw 'Installing the requirements failed.' }

        if (-not $NoStart) {
            Write-Host '==> Starting WhisperTyper (look for its icon in the system tray)'
            Start-Process -FilePath (Join-Path $venv 'Scripts\pythonw.exe') -ArgumentList "`"$(Join-Path $root 'run.py')`"" -WorkingDirectory $root
        }
        Write-Host 'Done. From now on: the Start menu entry, or whispertyper in a new terminal.' -ForegroundColor Green
    }
    catch {
        Write-Host $_.Exception.Message -ForegroundColor Yellow
        # Through iex there is no script to leave: exit would close the user's terminal.
        if ($ScriptFolder) { exit 1 }
    }
} $NoStart.IsPresent $PSScriptRoot $Python
