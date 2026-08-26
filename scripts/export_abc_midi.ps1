[CmdletBinding()]
param(
    [string[]]$Piece = @('n11op95_01'),
    [switch]$All,
    [string]$MuseScore = 'C:\Program Files\MuseScore 4\bin\MuseScore4.exe',
    [string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$scoreDirectory = Join-Path $projectRoot 'external\ABC\MS3'

if (-not (Test-Path -LiteralPath $MuseScore -PathType Leaf)) {
    throw "MuseScore 4 was not found at '$MuseScore'. Install MuseScore or pass -MuseScore with the path to MuseScore4.exe."
}
if (-not (Test-Path -LiteralPath $scoreDirectory -PathType Container)) {
    throw "ABC score directory was not found at '$scoreDirectory'."
}

if ($All) {
    $scores = @(Get-ChildItem -LiteralPath $scoreDirectory -Filter '*.mscx' | Sort-Object Name)
    if (-not $OutputDirectory) {
        $OutputDirectory = Join-Path $projectRoot 'results\midi'
    }
} else {
    $scores = @(
        foreach ($name in $Piece) {
            $stem = [IO.Path]::GetFileNameWithoutExtension($name)
            $path = Join-Path $scoreDirectory ($stem + '.mscx')
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "ABC score was not found: '$path'."
            }
            Get-Item -LiteralPath $path
        }
    )
    if (-not $OutputDirectory) {
        $OutputDirectory = $projectRoot
    }
}

$outputPath = [IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

foreach ($score in $scores) {
    $target = Join-Path $outputPath ($score.BaseName + '.mid')
    Write-Host "Exporting $($score.Name) -> $target"
    # MuseScore4.exe is a GUI-subsystem process and may return control to
    # PowerShell before conversion finishes when invoked with '&'. Wait for
    # the process explicitly so the output check is meaningful.
    $process = Start-Process -FilePath $MuseScore -ArgumentList @('-o', $target, $score.FullName) -Wait -PassThru
    if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $target -PathType Leaf)) {
        throw "MuseScore failed to export '$($score.FullName)'."
    }
}

Write-Host "Exported $($scores.Count) MIDI file(s) to $outputPath"
