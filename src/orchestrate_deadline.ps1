#!/usr/bin/env pwsh
# Deadline-driven orchestrator (2026-05-18).
# Runs jobs from BACKLOG.md in priority order until deadline.
# - GPU jobs and CPU jobs run in parallel (separate worker threads).
# - Each job's failure is logged but never stalls the queue.
# - After every job, the runner logs elapsed/remaining/queue_depth.
# - When budget hits or backlog empties, runs the analyzer + finalizer.
#
# Usage:
#   pwsh -File src/orchestrate_deadline.ps1 -DeadlineHours 48
#   pwsh -File src/orchestrate_deadline.ps1 -DeadlineHours 12 -BacklogFile BACKLOG.md

param(
    [double]$DeadlineHours = 48,
    [string]$BacklogFile   = "BACKLOG.md"
)

$ErrorActionPreference = "Continue"
$ProjectRoot = "C:\Users\jabir\Hacker_J\pingpong_aicup_2026"
$LogDir      = Join-Path $ProjectRoot "logs"
$Python      = "C:\Users\jabir\miniconda3\python.exe"

$OrchLog        = Join-Path $LogDir "orchestrate_deadline.log"
$FailedJobsLog  = Join-Path $LogDir "BACKLOG_FAILED.log"
$DoneJobsLog    = Join-Path $LogDir "BACKLOG_DONE.log"
$ValidatedLog   = Join-Path $LogDir "BACKLOG_VALIDATED.log"
$DoneMarker     = Join-Path $LogDir "deadline_orchestrator_DONE.marker"

$StartTime = Get-Date
$Deadline  = $StartTime.AddHours($DeadlineHours)

function Log-Msg {
    param([string]$Msg)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $OrchLog -Value "[$ts] $Msg"
    Write-Host "[$ts] $Msg"
}

function Get-Remaining-Hours {
    $r = ($Deadline - (Get-Date)).TotalHours
    return [math]::Round($r, 2)
}

function Get-Elapsed-Hours {
    $e = ((Get-Date) - $StartTime).TotalHours
    return [math]::Round($e, 2)
}

function Parse-Backlog {
    param([string]$Path)
    $jobs = @()
    $content = Get-Content -Path $Path -Raw
    # Match lines like: J001 | 1 | CPU |  87 | tag | cmd|with|pipes
    $lines = $content -split "`n" | Where-Object { $_ -match '^J\d{3}\s*\|' }
    foreach ($line in $lines) {
        $parts = $line -split '\|', 6
        if ($parts.Count -lt 6) { continue }
        $jobs += [PSCustomObject]@{
            Id      = $parts[0].Trim()
            Prio    = [int]$parts[1].Trim()
            Res     = $parts[2].Trim()
            EstMin  = [int]$parts[3].Trim()
            Tag     = $parts[4].Trim()
            PyArgs  = ($parts[5].Trim() -split '\|')
        }
    }
    return $jobs
}

function Is-Job-Done {
    param([string]$Tag)
    $oof = Join-Path $ProjectRoot "oof_predictions/${Tag}_oof_act.npy"
    return (Test-Path $oof)
}

function Run-Job {
    param([PSCustomObject]$Job)
    $log = Join-Path $LogDir "$($Job.Tag)_full.log"
    $errLog = "$log.err"
    Log-Msg "[$($Job.Id)] Launching $($Job.Tag) (RES=$($Job.Res), est=$($Job.EstMin)m) -> $log"
    try {
        $proc = Start-Process -FilePath $Python -ArgumentList $Job.PyArgs `
                              -WorkingDirectory $ProjectRoot `
                              -RedirectStandardOutput $log `
                              -RedirectStandardError $errLog `
                              -PassThru -NoNewWindow
        Log-Msg "[$($Job.Id)] PID=$($proc.Id) started"
        $proc.WaitForExit()
        $code = $proc.ExitCode
        if ($code -eq 0) {
            Add-Content -Path $DoneJobsLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $($Job.Id) $($Job.Tag) WALL=$([math]::Round(((Get-Date) - $proc.StartTime).TotalMinutes,1))m"
            Log-Msg "[$($Job.Id)] DONE rc=0"
        } else {
            Add-Content -Path $FailedJobsLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $($Job.Id) $($Job.Tag) rc=$code log=$log"
            Log-Msg "[$($Job.Id)] FAILED rc=$code (logged, continuing)"
        }
        return $code
    } catch {
        Add-Content -Path $FailedJobsLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $($Job.Id) $($Job.Tag) launch-exception: $_"
        Log-Msg "[$($Job.Id)] LAUNCH FAILED: $_"
        return -1
    }
}

# Reset markers
if (Test-Path $DoneMarker) { Remove-Item $DoneMarker -Force }

Log-Msg "==== Deadline orchestrator starting ===="
Log-Msg "  start=$($StartTime.ToString('yyyy-MM-dd HH:mm:ss'))"
Log-Msg "  deadline=$($Deadline.ToString('yyyy-MM-dd HH:mm:ss')) (${DeadlineHours}h)"
Log-Msg "  backlog=$BacklogFile"

$backlogPath = Join-Path $ProjectRoot $BacklogFile
if (-not (Test-Path $backlogPath)) {
    Log-Msg "FATAL: backlog file not found: $backlogPath"
    exit 1
}

$allJobs = Parse-Backlog -Path $backlogPath
Log-Msg "Backlog parsed: $($allJobs.Count) total jobs"

# Filter out already-done jobs
$pendingJobs = $allJobs | Where-Object { -not (Is-Job-Done -Tag $_.Tag) }
$skippedCount = $allJobs.Count - $pendingJobs.Count
Log-Msg "Already done: $skippedCount; pending: $($pendingJobs.Count)"

# Sort by priority (ascending: 1 = highest)
$pendingJobs = $pendingJobs | Sort-Object Prio, Id

# Split into GPU and CPU queues
$gpuQueue = [System.Collections.Generic.Queue[PSCustomObject]]::new()
$cpuQueue = [System.Collections.Generic.Queue[PSCustomObject]]::new()
foreach ($j in $pendingJobs) {
    if ($j.Res -eq "GPU") { $gpuQueue.Enqueue($j) }
    else { $cpuQueue.Enqueue($j) }
}
Log-Msg "GPU queue depth=$($gpuQueue.Count), CPU queue depth=$($cpuQueue.Count)"

# Direct dual-queue runner — alternates one tick on GPU side, one on CPU side.
# Same-process Start-Process for the Python child (proven pattern from Phase 1).
Log-Msg "Starting dual-queue loop (direct Start-Process, no Start-Job)"

$gpuArr = @($gpuQueue.ToArray())
$cpuArr = @($cpuQueue.ToArray())
$gpuIdx = 0
$cpuIdx = 0
$gpuProc = $null
$cpuProc = $null
$gpuCurJob = $null
$cpuCurJob = $null
$gpuStart = $null
$cpuStart = $null

function Start-Next-Job {
    param([string]$Lane, [object[]]$ArrRef, [ref]$IdxRef)
    while ($IdxRef.Value -lt $ArrRef.Count) {
        $j = $ArrRef[$IdxRef.Value]
        $IdxRef.Value++
        if (Is-Job-Done -Tag $j.Tag) {
            Log-Msg "[$Lane] [$($j.Id)] SKIP (already done) $($j.Tag)"
            continue
        }
        $remainingMin = ($script:Deadline - (Get-Date)).TotalMinutes
        if ($j.EstMin -gt $remainingMin) {
            Log-Msg "[$Lane] [$($j.Id)] SKIP est=$($j.EstMin)m > remaining=$([math]::Round($remainingMin,1))m"
            continue
        }
        $log = Join-Path $script:LogDir "$($j.Tag)_full.log"
        $errLog = "$log.err"
        try {
            $p = Start-Process -FilePath $script:Python -ArgumentList $j.PyArgs `
                               -WorkingDirectory $script:ProjectRoot `
                               -RedirectStandardOutput $log `
                               -RedirectStandardError $errLog `
                               -PassThru -NoNewWindow
            Log-Msg "[$Lane] [$($j.Id)] LAUNCH $($j.Tag) PID=$($p.Id) est=$($j.EstMin)m"
            return @{ Proc = $p; Job = $j; Start = (Get-Date) }
        } catch {
            Add-Content -Path $script:FailedJobsLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $($j.Id) $($j.Tag) launch-exception: $_"
            Log-Msg "[$Lane] [$($j.Id)] LAUNCH EXCEPTION: $_"
        }
    }
    return $null
}

function Validate-Artifact {
    param([string]$Lane, [string]$JobId, [string]$Tag)
    Log-Msg "[$Lane] [$JobId] VALIDATE artifact for $Tag"
    try {
        $p = Start-Process -FilePath $script:Python `
            -ArgumentList @("-u", "src/validate_oof_artifact.py",
                            "--tag", $Tag,
                            "--log-file", $script:ValidatedLog) `
            -WorkingDirectory $script:ProjectRoot `
            -RedirectStandardOutput (Join-Path $script:LogDir "validate_${Tag}.log") `
            -RedirectStandardError  (Join-Path $script:LogDir "validate_${Tag}.log.err") `
            -PassThru -NoNewWindow
        $p.WaitForExit()
        $vcode = $p.ExitCode
        $verdict = switch ($vcode) {
            0 { "ELIGIBLE" }
            1 { "INELIGIBLE" }
            2 { "ELIGIBLE_WITH_WARNINGS" }
            default { "UNKNOWN($vcode)" }
        }
        Log-Msg "[$Lane] [$JobId] VALIDATE $Tag => $verdict"
    } catch {
        Log-Msg "[$Lane] [$JobId] VALIDATE EXCEPTION: $_"
    }
}

function Reap-If-Done {
    param([string]$Lane, $ProcInfo)
    if ($null -eq $ProcInfo) { return $null }
    if (-not $ProcInfo.Proc.HasExited) { return $ProcInfo }
    $code = $ProcInfo.Proc.ExitCode
    $wallMin = [math]::Round(((Get-Date) - $ProcInfo.Start).TotalMinutes, 1)
    $j = $ProcInfo.Job
    if ($code -eq 0) {
        Add-Content -Path $script:DoneJobsLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $($j.Id) $($j.Tag) WALL=${wallMin}m"
        Log-Msg "[$Lane] [$($j.Id)] DONE rc=0 wall=${wallMin}m (est $($j.EstMin)m)"
        # User directive 2026-05-18: validate artifact after each successful job.
        Validate-Artifact -Lane $Lane -JobId $j.Id -Tag $j.Tag
    } else {
        Add-Content -Path $script:FailedJobsLog -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $($j.Id) $($j.Tag) rc=$code"
        Log-Msg "[$Lane] [$($j.Id)] FAILED rc=$code wall=${wallMin}m (continuing)"
    }
    return $null
}

# Initial launches
$gpuRefIdx = [ref]$gpuIdx
$cpuRefIdx = [ref]$cpuIdx
$gpuPi = Start-Next-Job -Lane "GPU" -ArrRef $gpuArr -IdxRef $gpuRefIdx
$cpuPi = Start-Next-Job -Lane "CPU" -ArrRef $cpuArr -IdxRef $cpuRefIdx
$gpuIdx = $gpuRefIdx.Value
$cpuIdx = $cpuRefIdx.Value

$lastStatusLog = Get-Date
while ($true) {
    Start-Sleep -Seconds 30
    $gpuPi = Reap-If-Done -Lane "GPU" -ProcInfo $gpuPi
    $cpuPi = Reap-If-Done -Lane "CPU" -ProcInfo $cpuPi

    if ($null -eq $gpuPi) {
        $gpuRefIdx = [ref]$gpuIdx
        $gpuPi = Start-Next-Job -Lane "GPU" -ArrRef $gpuArr -IdxRef $gpuRefIdx
        $gpuIdx = $gpuRefIdx.Value
    }
    if ($null -eq $cpuPi) {
        $cpuRefIdx = [ref]$cpuIdx
        $cpuPi = Start-Next-Job -Lane "CPU" -ArrRef $cpuArr -IdxRef $cpuRefIdx
        $cpuIdx = $cpuRefIdx.Value
    }

    # Periodic status (every 30 min)
    if (((Get-Date) - $lastStatusLog).TotalMinutes -ge 30) {
        Log-Msg "STATUS elapsed=$(Get-Elapsed-Hours)h remaining=$(Get-Remaining-Hours)h gpu_idx=$gpuIdx/$($gpuArr.Count) cpu_idx=$cpuIdx/$($cpuArr.Count)"
        $lastStatusLog = Get-Date
    }

    # Exit when both queues exhausted AND no in-flight jobs
    if ($null -eq $gpuPi -and $null -eq $cpuPi -and $gpuIdx -ge $gpuArr.Count -and $cpuIdx -ge $cpuArr.Count) {
        Log-Msg "Both queues exhausted, no in-flight jobs. Done."
        break
    }

    # Exit at deadline
    if ((Get-Date) -ge $Deadline) {
        Log-Msg "Deadline reached. Letting in-flight jobs finish."
        if ($null -ne $gpuPi) { $gpuPi.Proc.WaitForExit() | Out-Null; $gpuPi = Reap-If-Done -Lane "GPU" -ProcInfo $gpuPi }
        if ($null -ne $cpuPi) { $cpuPi.Proc.WaitForExit() | Out-Null; $cpuPi = Reap-If-Done -Lane "CPU" -ProcInfo $cpuPi }
        break
    }
}

Log-Msg "==== Both queues exhausted / deadline reached. Running avg builder only ===="

# Build derived avg components (idempotent — skips any missing sources).
# NOTE: user directive 2026-05-18 — DO NOT run the submission-generating
# analyzer here. Per-job validation already ran inside Reap-If-Done; the
# analyzer + LB submission step waits for explicit human review.
$avgScript = Join-Path $ProjectRoot "src\_build_avg.py"
if (Test-Path $avgScript) {
    $avgLog = Join-Path $LogDir "deadline_build_avg.log"
    $proc = Start-Process -FilePath $Python -ArgumentList @("-u", $avgScript) `
            -WorkingDirectory $ProjectRoot -RedirectStandardOutput $avgLog `
            -RedirectStandardError "$avgLog.err" -PassThru -NoNewWindow
    $proc.WaitForExit()
    Log-Msg "Avg builder exit=$($proc.ExitCode) (see $avgLog)"
}

Log-Msg "==== Skipping auto-submission analyzer per user directive ===="
Log-Msg "When ready, manually run: python -u src/analyze_phase3.py"
Log-Msg "Per-job validation summary: see $ValidatedLog"

"Deadline orchestrator finished at $(Get-Date). NO submissions generated." | `
    Out-File -FilePath $DoneMarker -Encoding utf8
Log-Msg "==== Deadline orchestrator DONE ===="
