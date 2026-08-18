# Run as ADMIN to arm vacation mode. DISARM: schtasks /delete /tn GpuWatchdog /f
#                                            schtasks /delete /tn GpuBootServe /f
schtasks /create /f /tn GpuWatchdog /sc minute /mo 5 /rl highest /ru SYSTEM `
  /tr "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\egerm\gpu-watchdog.ps1"
schtasks /create /f /tn GpuBootServe /sc onstart /delay 0002:00 /rl highest /ru egerm `
  /tr "wsl -d Ubuntu -u erol bash /home/erol/.config/llama-tcq/start-long-38.sh"
Write-Host "VACATION MODE ARMED. Disarm with the two /delete commands in this file."
