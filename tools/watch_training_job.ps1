param(
    [Parameter(Mandatory = $true)]
    [string]$JobId,
    [string]$BaseUrl = "http://127.0.0.1:3004",
    [string]$RepoRoot = "C:\Users\linco\OneDrive\Desktop\Mangio-RVC-Fork-main",
    [int]$PollSeconds = 3
)

$ErrorActionPreference = "SilentlyContinue"

function Get-LatestGuidedSummary {
    param(
        [string]$JobRoot
    )

    $datasetRoot = Join-Path $JobRoot "pipa-build\_guided_svs_dataset"
    $featuresRoot = Join-Path $datasetRoot "features"
    $historyPath = Join-Path $JobRoot "pipa-build\guided_regeneration\guided_regeneration_history.json"
    $reportPath = Join-Path $JobRoot "pipa-build\guided_regeneration\guided_regeneration_report.json"

    $featureDirs = @()
    if (Test-Path $featuresRoot) {
        $featureDirs = Get-ChildItem $featuresRoot -Directory | Sort-Object LastWriteTime -Descending
    }

    $summary = [ordered]@{
        feature_count = $featureDirs.Count
        latest_feature = ""
        latest_feature_time = ""
        epoch_line = ""
        report_line = ""
    }

    if ($featureDirs.Count -gt 0) {
        $summary.latest_feature = $featureDirs[0].Name
        $summary.latest_feature_time = $featureDirs[0].LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss")
    }

    if (Test-Path $historyPath) {
        try {
            $history = Get-Content $historyPath -Raw | ConvertFrom-Json
            if ($history -and $history.Count -gt 0) {
                $last = $history[-1]
                $summary.epoch_line = "epoch=$($last.epoch) train_l1=$($last.train_l1) val_l1=$($last.val_l1) best=$($last.best_val_l1)"
            }
        } catch {
        }
    }

    if (Test-Path $reportPath) {
        try {
            $report = Get-Content $reportPath -Raw | ConvertFrom-Json
            $summary.report_line = "best_val_l1=$($report.best_val_l1) last_epoch=$($report.last_epoch) stopped_early=$($report.stopped_early)"
        } catch {
        }
    }

    return $summary
}

$jobRoot = Join-Path $RepoRoot ("training-runs\" + $JobId)
$statusUrl = ($BaseUrl.TrimEnd("/") + "/api/training/jobs/" + $JobId)

Write-Host ""
Write-Host ("Watching training job " + $JobId) -ForegroundColor Cyan
Write-Host ("Status endpoint: " + $statusUrl) -ForegroundColor DarkGray
Write-Host ("Job root: " + $jobRoot) -ForegroundColor DarkGray
Write-Host ""

$lastSignature = ""

while ($true) {
    $timestamp = Get-Date -Format "HH:mm:ss"
    $payload = $null

    try {
        $payload = Invoke-RestMethod -Uri $statusUrl -TimeoutSec 10
    } catch {
        Write-Host ("[" + $timestamp + "] waiting for status endpoint...") -ForegroundColor Yellow
        Start-Sleep -Seconds $PollSeconds
        continue
    }

    $guided = Get-LatestGuidedSummary -JobRoot $jobRoot
    $signature = @(
        $payload.status,
        $payload.stage,
        $payload.progress,
        $payload.message,
        $payload.log_tail,
        $guided.feature_count,
        $guided.latest_feature,
        $guided.epoch_line,
        $guided.report_line
    ) -join "|"

    if ($signature -ne $lastSignature) {
        Clear-Host
        Write-Host ("Training watcher - " + $JobId) -ForegroundColor Cyan
        Write-Host ("Updated: " + $timestamp) -ForegroundColor DarkGray
        Write-Host ""
        Write-Host ("Status   : " + $payload.status)
        Write-Host ("Stage    : " + $payload.stage)
        Write-Host ("Progress : " + $payload.progress + "%")
        Write-Host ("Message  : " + $payload.message)
        if ($payload.log_tail) {
            Write-Host ("Detail   : " + $payload.log_tail)
        }
        Write-Host ""
        Write-Host ("SVS clips built : " + $guided.feature_count)
        if ($guided.latest_feature) {
            Write-Host ("Latest clip     : " + $guided.latest_feature + " @ " + $guided.latest_feature_time)
        }
        if ($guided.epoch_line) {
            Write-Host ("Epoch metrics   : " + $guided.epoch_line) -ForegroundColor Green
        }
        if ($guided.report_line) {
            Write-Host ("Checkpoint info : " + $guided.report_line) -ForegroundColor Green
        }
        if ($payload.error) {
            Write-Host ""
            Write-Host "Error:" -ForegroundColor Red
            Write-Host $payload.error -ForegroundColor Red
        }
        Write-Host ""
        Write-Host "Press Ctrl+C in this window to stop watching." -ForegroundColor DarkGray
        $lastSignature = $signature
    }

    if ($payload.status -in @("completed", "failed", "stopped")) {
        Write-Host ""
        Write-Host ("Training ended with status: " + $payload.status) -ForegroundColor Cyan
        break
    }

    Start-Sleep -Seconds $PollSeconds
}
