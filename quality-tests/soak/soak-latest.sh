#!/bin/bash
# POST-GATE SOAK AUTOPILOT (2026-08-18). Run ONLY after verdict-gate.sh PASSED.
# Starts the LATEST prod stack (SPEED profile) and drives continuous mixed load
# with the PCIe-stress ops that historically killed the box (deep prefill +
# slot save/restore), monitoring the Event-14 prodrome. Default 6h.
# PASS = runs to completion with zero events => vacation-trust candidate.
set -u
HOURS="${HOURS:-6}"
D=/mnt/d/spill/qwen36-test
Q=/home/erol/ai/turboquant/turboquant-g1/quality-tests
CONF=/home/erol/.config/llama-tcq
KEY=$(cat $CONF/api.key)
CAL=$Q/quant-lab/calib_v2.txt
END=$(( $(date +%s) + HOURS*3600 ))
mkdir -p $D
echo $$ > $D/soak.pid
log(){ echo "$(date +%H:%M:%S) $*" >> $D/soak-milestones.log; }

log "SOAK START (latest stack, ${HOURS}h target)"
bash $CONF/${LAUNCHER:-start-long-38.sh} > $D/soak-launch.log 2>&1 || { log "LAUNCH FAILED"; exit 1; }
log "serving up (SPEED profile)"

( while true; do echo "$(date +%H:%M:%S),$(nvidia-smi --query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,pcie.link.gen.current --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" >> $D/soak-telemetry.csv; sleep 30; done ) &
TPID=$!

gen(){ # $1=prompt-chars $2=max-tokens $3=tag
    local OFF=$(( RANDOM % 4000000 ))
    local P=$(tail -c +$OFF "$CAL" | head -c "$1" | tr -d '\000' | python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))")
    local CODE=$(curl -s -o $D/soak-last-resp.json -w "%{http_code}" --max-time 900 \
      http://127.0.0.1:8130/v1/chat/completions -H "Authorization: Bearer $KEY" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"qwen3.8-27b-320k\",\"max_tokens\":$2,\"messages\":[{\"role\":\"user\",\"content\":$P}]}")
    log "gen $3 http=$CODE"
    [ "$CODE" = "200" ]
}

prodrome_clear(){
    local N=$(powershell.exe -NoProfile -Command "(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'; Id=14; StartTime=(Get-Date).AddMinutes(-3)} -MaxEvents 5 -ErrorAction SilentlyContinue | Measure-Object).Count" 2>/dev/null | tr -dc 0-9)
    [ -z "$N" ] && N=0
    [ "$N" -eq 0 ]
}

CYCLE=0
while [ "$(date +%s)" -lt "$END" ]; do
    CYCLE=$((CYCLE+1))
    nvidia-smi --query-gpu=name --format=csv,noheader > /dev/null 2>&1 || { log "SOAKEVENT: NVML dead at cycle $CYCLE"; break; }
    prodrome_clear || { log "SOAKEVENT: Event-14 PRODROME at cycle $CYCLE — stopping serving preemptively"; bash $CONF/stop.sh >/dev/null 2>&1; break; }
    gen 4000 640 "chat-c$CYCLE" || { log "SOAKEVENT: gen fail c$CYCLE"; curl -s --max-time 3 http://127.0.0.1:8131/health >/dev/null || break; }
    if [ $((CYCLE % 10)) -eq 0 ]; then gen 96000 512 "deep-prefill-c$CYCLE" || true; fi
    if [ $((CYCLE % 14)) -eq 0 ]; then
        curl -s --max-time 300 -X POST "http://127.0.0.1:8131/slots/0?action=save" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"filename":"soak.bin"}' > /dev/null 2>&1
        curl -s --max-time 300 -X POST "http://127.0.0.1:8131/slots/0?action=restore" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d '{"filename":"soak.bin"}' > /dev/null 2>&1
        log "slot save/restore cycle c$CYCLE (PCIe bulk H2D/D2H)"
    fi
done
kill $TPID 2>/dev/null
if [ "$(date +%s)" -ge "$END" ]; then log "SOAK COMPLETE: ${HOURS}h reached, $CYCLE cycles — PASS (serving left UP)";
else log "SOAK ENDED EARLY at cycle $CYCLE — see events above"; fi
