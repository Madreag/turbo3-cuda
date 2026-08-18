# GPU watchdog (VACATION MODE — OPT-IN, violates no-autostart law by design;
# arm only with user's explicit word). Runs as Scheduled Task every 5 min.
# 2 consecutive nvidia-smi failures => log + reboot. Boot task restarts serving.
$log = "C:\Users\egerm\vacation-watchdog.log"
$stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
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
