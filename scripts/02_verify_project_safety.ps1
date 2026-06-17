# ============================================================
# VERIFY PROJECT SAFETY BEFORE COMMITTING
# ============================================================

# ============================================================
# STOP ON POWERSHELL ERRORS
# ============================================================
$ErrorActionPreference = "Stop"

# ============================================================
# PREVENT POWERSHELL 7 NATIVE-COMMAND NONZERO EXITS FROM THROWING BEFORE $LASTEXITCODE CAN BE READ
# ============================================================
$PSNativeCommandUseErrorActionPreference = $false

# ============================================================
# DEFINE THE PROJECT ROOT AS THE PARENT OF THE SCRIPTS FOLDER
# ============================================================
$ProjectRoot = Split-Path -Parent $PSScriptRoot

# ============================================================
# MOVE TO THE PROJECT ROOT SO GIT COMMANDS USE THE CORRECT REPOSITORY
# ============================================================
Set-Location $ProjectRoot

# ============================================================
# DEFINE LOCAL SECRET FILES THAT MUST NEVER BE TRACKED
# ============================================================
$SensitiveLocalPaths = @(
    ".env",
    "config/config.local.json",
    "config.local.json",
    "azure-credentials.json"
)

# ============================================================
# VERIFY EACH SENSITIVE PATH IS NOT TRACKED BY GIT, WHETHER OR NOT THE FILE EXISTS LOCALLY
# ============================================================
foreach ($SensitivePath in $SensitiveLocalPaths) {
    $TrackedOutput = & git ls-files -- $SensitivePath

    if ($TrackedOutput) {
        Write-Host "Local secret file is already tracked by Git: $SensitivePath" -ForegroundColor Red
        exit 1
    }
}

# ============================================================
# VERIFY EACH EXISTING LOCAL SECRET FILE IS IGNORED BY GIT
# ============================================================
foreach ($SensitivePath in $SensitiveLocalPaths) {
    if (Test-Path (Join-Path $ProjectRoot $SensitivePath)) {
        & git check-ignore -q -- $SensitivePath
        $IsIgnored = ($LASTEXITCODE -eq 0)

        if (-not $IsIgnored) {
            Write-Host "Local secret file is NOT ignored by .gitignore: $SensitivePath" -ForegroundColor Red
            exit 1
        }
    }
}

# ============================================================
# DEFINE FILES THAT ARE SAFE EXAMPLES OR THE SCANNER ITSELF
# ============================================================
$ExcludedFileNames = @(
    ".env.example",
    "config.example.json",
    "02_verify_project_safety.ps1"
)

# ============================================================
# DEFINE TEXT FILE EXTENSIONS TO SCAN
# ============================================================
$AllowedExtensions = @(
    ".py",
    ".ps1",
    ".yml",
    ".yaml",
    ".json",
    ".toml",
    ".txt",
    ".md",
    ".env",
    ".ini",
    ".cfg"
)

# ============================================================
# DEFINE REAL-LOOKING SECRET PATTERNS
# ============================================================
$Patterns = @(
    "sk-[A-Za-z0-9_\-]{20,}",
    "gh[pousr]_[A-Za-z0-9_]{30,}",
    "(?i)(password|secret|token|api[_-]?key)\s*[:=]\s*['""][^'""]{12,}['""]",
    "(?i)clientSecret\s*[:=]\s*['""][^'""]{12,}['""]"
)

# ============================================================
# DEFINE SAFE PLACEHOLDER OR CI SECRET REFERENCE PATTERNS
# ============================================================
$SafeReferencePatterns = @(
    "\$\{\{\s*secrets\.",
    "\$\{\{\s*env\.",
    "REPLACE_WITH",
    "YOUR-",
    "DO_NOT_COMMIT",
    "placeholder",
    "example",
    "local_only"
)

# ============================================================
# COLLECT TRACKED FILES
# ============================================================
$TrackedFiles = & git ls-files

# ============================================================
# COLLECT UNTRACKED FILES THAT ARE NOT IGNORED BY GIT
# ============================================================
$UntrackedNonIgnoredFiles = & git ls-files --others --exclude-standard

# ============================================================
# COMBINE TRACKED AND COMMIT-CANDIDATE FILES
# ============================================================
$RelativeFiles = @($TrackedFiles + $UntrackedNonIgnoredFiles) |
    Where-Object { $_ -and $_.Trim().Length -gt 0 } |
    Sort-Object -Unique

# ============================================================
# FILTER CANDIDATE FILES BY EXTENSION AND EXCLUDED SAFE EXAMPLE NAMES
# ============================================================
$Files = foreach ($RelativeFile in $RelativeFiles) {
    $FullPath = Join-Path $ProjectRoot $RelativeFile
    $FileName = Split-Path $FullPath -Leaf
    $Extension = [System.IO.Path]::GetExtension($FullPath)

    if ((Test-Path $FullPath) -and
        ($AllowedExtensions -contains $Extension) -and
        (-not ($ExcludedFileNames -contains $FileName))) {
        Get-Item $FullPath
    }
}

# ============================================================
# SCAN FILES FOR SECRET-LIKE VALUES
# ============================================================
$RawFindings = foreach ($File in $Files) {
    Select-String -Path $File.FullName -Pattern $Patterns -AllMatches -ErrorAction SilentlyContinue
}

# ============================================================
# REMOVE SAFE PLACEHOLDER REFERENCES FROM FINDINGS
# ============================================================
$Findings = foreach ($Finding in $RawFindings) {
    $IsSafeReference = $false

    foreach ($SafePattern in $SafeReferencePatterns) {
        if ($Finding.Line -match $SafePattern) {
            $IsSafeReference = $true
        }
    }

    if (-not $IsSafeReference) {
        $Finding
    }
}

# ============================================================
# FAIL WHEN SECRET-LIKE VALUES ARE FOUND
# ============================================================
if ($Findings) {
    Write-Host "Potential secret-like values were found. Review before committing." -ForegroundColor Red
    $Findings | ForEach-Object {
        Write-Host "$($_.Path):$($_.LineNumber): $($_.Line)" -ForegroundColor Yellow
    }
    exit 1
}

# ============================================================
# PASS WHEN NO SECRET-LIKE VALUES ARE FOUND
# ============================================================
Write-Host "Safety check passed. No tracked or commit-candidate secret-like values found." -ForegroundColor Green
exit 0
