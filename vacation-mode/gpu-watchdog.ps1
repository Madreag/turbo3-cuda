# GPU watchdog (VACATION MODE — OPT-IN, violates no-autostart law by design;
# arm only with user's explicit word). Runs as Scheduled Task every 5 min.
# 2 consecutive nvidia-smi failures => log + reboot. Boot task restarts serving.
# 2026-08-18 upgrade: also polls nvlddmkm Event-14 CMDre bursts — the measured
# pre-death prodrome (~5 min lead on 0x116). On a fresh burst: log LOUDLY and
# stop serving cleanly BEFORE the TDR takes the whole machine down.
$log = "C:\Users\egerm\vacation-watchdog.log"
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# --- prodrome check: nvlddmkm Event 14 in the last 6 minutes ---
try {
    $ev14 = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'; Id=14; StartTime=(Get-Date).AddMinutes(-6)} -MaxEvents 3 -ErrorAction SilentlyContinue
    if ($ev14) {
        Add-Content $log "$stamp PRODROME: nvlddmkm Event-14 burst detected ($($ev14.Count)+ in 6min) - stopping serving preemptively"
        # Stop the WSL serving stack cleanly before the link failure escalates.
        wsl.exe -e bash -lc "bash ~/.config/llama-tcq/stop.sh" 2>$null
        Add-Content $log "$stamp serving stopped; letting link quiesce (no reboot on prodrome alone)"
    }
} catch {}

# --- adapter-alive check (original logic) ---
$ok = $false
try { & nvidia-smi.exe --query-gpu=name --format=csv,noheader 2>$null | Out-Null; $ok = ($LASTEXITCODE -eq 0) } catch {}
$flag = "C:\Users\egerm\gpu-fail-flag"
if ($ok) { Remove-Item $flag -ErrorAction SilentlyContinue; Add-Content $log "$stamp OK"; exit 0 }
if (Test-Path $flag) {
    Add-Content $log "$stamp SECOND FAILURE - REBOOTING"
    Remove-Item $flag
    shutdown /r /t 60 /c "GPU watchdog: adapter lost, auto-recovery reboot"
} else {
    New-Item $flag -ItemType File | Out-Null
    Add-Content $log "$stamp first failure, flagged"
}
