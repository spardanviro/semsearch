# semsearch installer for Windows (Windows PowerShell 5.1 or PowerShell 7).
#
#   powershell -ExecutionPolicy ByPass -c "irm https://github.com/spardanviro/semsearch/releases/latest/download/install.ps1 | iex"
#
# or, from a clone:  powershell -ExecutionPolicy ByPass -File scripts\install.ps1
#
# Steps; each one that downloads something or changes your setup asks first:
#   1. uv (Python tool manager) from astral.sh, if missing
#   2. semsearch with dependencies pinned to uv.lock (PyTorch CUDA build: about 3.5 GB on disk)
#   3. `semsearch doctor`: checks GPU, driver, model, disk, Claude Code
#   4. the embedding model, about 1.5 GB (`semsearch download-model`)
#   5. registration with Claude Code (`semsearch install-claude`)
#
# Environment: SEMSEARCH_INSTALL_REF     branch, tag or commit to install (default: DEFAULT_REF below)
#              SEMSEARCH_INSTALL_SOURCE  install from this local folder instead of GitHub
#              SEMSEARCH_INSTALL_YES=1   answer yes to every question

function Install-Semsearch {
    # a function, not a script body: `irm | iex` runs in the caller's session, where `exit`
    # would close the user's window.
    # 'Continue': under 'Stop', Windows PowerShell 5.1 turns uv's progress output on stderr into
    # a terminating error whenever stderr is redirected. Native commands are judged by their
    # exit code instead, and cmdlets that must not fail say -ErrorAction Stop.
    $ErrorActionPreference = 'Continue'
    $ProgressPreference = 'SilentlyContinue'    # Invoke-WebRequest is very slow with progress
    $repo = 'spardanviro/semsearch'
    $defaultRef = 'main'   # DEFAULT_REF: a release's copy of this script names its own tag
    $ref = if ($env:SEMSEARCH_INSTALL_REF) { $env:SEMSEARCH_INSTALL_REF } else { $defaultRef }
    $autoYes = $env:SEMSEARCH_INSTALL_YES -eq '1'

    function Write-Step([string]$text) { Write-Host ''; Write-Host "==> $text" -ForegroundColor Cyan }
    function Write-Note([string]$text) { Write-Host "    $text" }
    function Confirm-Step([string]$question) {
        if ($autoYes) { return $true }
        try { $answer = Read-Host "$question [y/N]" } catch { return $false }   # no console
        return ($answer.Trim().ToLower() -eq 'y')
    }
    function Invoke-Checked([string]$exe, [string[]]$argList) {
        & $exe @argList
        if ($LASTEXITCODE -ne 0) { throw "$exe $($argList -join ' ') failed (exit code $LASTEXITCODE)" }
    }
    function Test-SemsearchSource([string]$dir) {
        $toml = Join-Path $dir 'pyproject.toml'
        return (Test-Path (Join-Path $dir 'uv.lock')) -and (Test-Path $toml) -and
               (Select-String -Path $toml -Pattern '^name = "semsearch"$' -Quiet)
    }

    Write-Host 'semsearch installer: local semantic code search for Claude Code' -ForegroundColor Green
    Write-Note 'Needs about 5 GB of disk: PyTorch about 3.5 GB, the model about 1.5 GB.'
    Write-Note 'An NVIDIA GPU with driver 580 or newer makes indexing much faster (optional).'

    # --- 1. uv ---------------------------------------------------------------------------
    Write-Step 'Checking uv'
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Note 'uv is not installed. It comes from its official installer at https://astral.sh/uv'
        if (-not (Confirm-Step 'Download and install uv now?')) {
            Write-Note 'Stopped. Install uv yourself (https://docs.astral.sh/uv/) and run this again.'
            return
        }
        # this PowerShell's own executable, not whatever `powershell` PATH finds first
        $shell = if ($PSVersionTable.PSEdition -eq 'Core') { 'pwsh.exe' } else { 'powershell.exe' }
        $env:UV_NO_MODIFY_PATH = '1'      # PATH is changed below, and only after asking
        try {
            Invoke-Checked (Join-Path $PSHOME $shell) @('-NoProfile', '-ExecutionPolicy', 'ByPass',
                '-Command', 'irm https://astral.sh/uv/install.ps1 | iex')
        } finally {
            Remove-Item Env:UV_NO_MODIFY_PATH -ErrorAction SilentlyContinue
        }
        $env:Path = "$HOME\.local\bin;$env:Path"
    }
    $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCommand) { throw 'uv was installed but cannot be found; open a new terminal and run this again' }
    $uv = $uvCommand.Source
    Write-Note (& $uv --version)

    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("semsearch-install-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $tmp -ErrorAction Stop | Out-Null
    try {
        # --- 2. semsearch ---------------------------------------------------------------
        Write-Step 'Installing semsearch'
        Write-Note 'The first install downloads PyTorch with CUDA (about 3.5 GB on disk); this takes a while.'
        if (-not (Confirm-Step 'Install semsearch now?')) {
            Write-Note 'Stopped. Nothing was installed.'
            return
        }
        $src = $env:SEMSEARCH_INSTALL_SOURCE
        if (-not $src -and $PSScriptRoot) {
            $parent = Join-Path $PSScriptRoot '..'
            if (Test-SemsearchSource $parent) { $src = (Resolve-Path $parent).Path }
        }
        if ($src) {
            Write-Note "from $src"
        } else {
            $url = "https://github.com/$repo/archive/$ref.zip"
            Write-Note "from $url"
            $zip = Join-Path $tmp 'src.zip'
            Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing -ErrorAction Stop
            Expand-Archive -Path $zip -DestinationPath (Join-Path $tmp 'src') -ErrorAction Stop
            $src = (Get-ChildItem -Directory (Join-Path $tmp 'src') | Select-Object -First 1).FullName
        }
        if (-not (Test-SemsearchSource $src)) { throw "$src is not a semsearch source folder" }

        $constraints = Join-Path $tmp 'constraints.txt'
        Invoke-Checked $uv @('export', '--project', $src, '--frozen', '--no-dev', '--no-hashes',
                             '--no-emit-project', '--quiet', '-o', $constraints)
        Invoke-Checked $uv @('tool', 'install', $src, '--python', '3.12',
                             '--reinstall-package', 'semsearch', '--constraints', $constraints)
        $bin = (& $uv tool dir --bin).Trim()
        $exe = Join-Path $bin 'semsearch.exe'
        if (-not (Test-Path $exe)) { throw "semsearch was installed but $exe is missing" }
        if (-not (($env:Path -split ';') -contains $bin)) {
            $env:Path = "$bin;$env:Path"
            Write-Note "$bin is not on your PATH, so new terminals will not find ``semsearch``."
            if (Confirm-Step 'Add it to your user PATH (uv tool update-shell)?') {
                & $uv tool update-shell
            }
        }

        # --- 3. doctor ------------------------------------------------------------------
        Write-Step 'Checking the environment (semsearch doctor)'
        & $exe doctor --offline

        # --- 4. model -------------------------------------------------------------------
        Write-Step 'Embedding model'
        $modelArgs = @('download-model')
        if ($autoYes) { $modelArgs += '--yes' }
        & $exe @modelArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Note 'The model is not downloaded; the first search downloads it, or run `semsearch download-model` later.'
        }

        # --- 5. Claude Code -------------------------------------------------------------
        Write-Step 'Registering with Claude Code'
        $claudeArgs = @('install-claude')
        if ($autoYes) { $claudeArgs += '--yes' }
        & $exe @claudeArgs
        Write-Host ''
        if ($LASTEXITCODE -eq 0) {
            Write-Host 'Done. Open a new Claude Code session and ask about your code as usual.' -ForegroundColor Green
        } else {
            Write-Host 'semsearch is installed but not registered with Claude Code yet.' -ForegroundColor Yellow
            Write-Note 'Run `semsearch install-claude` when you are ready.'
        }
        Write-Note 'Check the setup any time with: semsearch doctor'
    } finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }
}

try {
    Install-Semsearch
} catch {
    Write-Host ''
    Write-Host "Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host 'Claude Code is registered only in the last step, so its settings were not changed.'
    Write-Host 'Fix the problem above and run the installer again.'
}
