param(
    [string]$RunName = "multimodal_model",
    [string]$DataPath = "data/esol.csv,data/freesolv.csv,data/lipo.csv",
    [string]$Device = "cuda",
    [string]$TrainingMode = "predictor",
    [string]$PythonEnv = "",
    [switch]$UseWandb = $true,
    [string]$WandbProject = "pruebas"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$pythonCmd = $null
$pythonArgs = @()

if ($PythonEnv) {
    $pythonCmd = "conda"
    $pythonArgs = @("run", "-n", $PythonEnv, "python")
    Write-Host "Using requested conda environment: $PythonEnv"
}
elseif (Get-Command conda -ErrorAction SilentlyContinue) {
    $activeEnv = $env:CONDA_DEFAULT_ENV
    if ($activeEnv -and $activeEnv -ne "base") {
        $pythonCmd = "conda"
        $pythonArgs = @("run", "-n", $activeEnv, "python")
        Write-Host "Using active conda environment: $activeEnv"
    }
    else {
        $condaEnv = "thesis"
        $condaPrefix = "C:\Users\snowo\miniconda3\envs\$condaEnv"
        if (Test-Path $condaPrefix) {
            $pythonCmd = "conda"
            $pythonArgs = @("run", "-n", $condaEnv, "python")
            Write-Host "Using conda environment: $condaEnv"
        }
    }
}

if (-not $pythonCmd) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $pythonCmd = "py"
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $pythonCmd = "python"
    }
    else {
        throw "No Python launcher was found. Install Python or make sure 'python' is available in PATH."
    }
}

$arguments = @(
    "train.py",
    "--data-path", $DataPath,
    "--task", "regression",
    "--training-mode", $TrainingMode,
    "--graph-backbone", "gin",
    "--language-backbone", "chemberta",
    "--fusion", "concat",
    "--hidden-dim", "256",
    "--num-layers", "3",
    "--num-heads", "8",
    "--dropout", "0.3",
    "--batch-size", "32",
    "--epochs", "50",
    "--lr", "0.001",
    "--weight-decay", "0.0001",
    "--seed", "2025",
    "--device", $Device,
    "--wandb-run-name", $RunName
)

if ($UseWandb) {
    $arguments += @("--use-wandb", "--wandb-project", $WandbProject)
}

$launchCmd = @($pythonCmd) + $pythonArgs + $arguments
Write-Host "Launching training with: $($launchCmd -join ' ')"
try {
    & $pythonCmd @pythonArgs @arguments
}
catch {
    Write-Host "Training failed. If you still see torch/torch_geometric errors, create or activate a working environment first." -ForegroundColor Yellow
    throw
}
