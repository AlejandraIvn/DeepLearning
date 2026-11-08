param(
    [switch]$Cpu,
    [ValidateRange(1, 65535)][int]$Port = 8000
)

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    [Console]::Error.WriteLine('Falta .venv\Scripts\python.exe. Sigue la instalacion inicial del README.')
    exit 1
}

$demoArgs = @('-m', 'backend.demo', '--port', $Port)
if ($Cpu) { $demoArgs += '--cpu' }

Push-Location -LiteralPath $PSScriptRoot
try {
    & $python @demoArgs
    $demoExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $demoExitCode
