#!/bin/bash
# Text-equivalence gate: NEW (sync build, 3 picks) vs PROD binary, greedy, byte-compare.
set -u
OUT=/home/erol/ai/turboquant/turboquant-g1/quality-tests/slotvision
MODEL=/home/erol/ai/turboquant/models/qwen38/Qwen3.8-27B-Q6_K.gguf
PROMPT='Explain in exactly three sentences why the sky appears blue.'
run_arm () { # $1=binary $2=outfile
    setsid nohup "$1" -m "$MODEL" --spec-type none -ctk turbo4 -ctv turbo4 \
      -fa on -ngl 99 -c 4096 --no-context-shift --jinja --reasoning-format none \
      --parallel 1 -b 512 -ub 512 --host 127.0.0.1 --port 8233 \
      < /dev/null > "$OUT/eq-server.log" 2>&1 &
    local pid=$!
    for i in $(seq 1 120); do
        code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8233/health 2>/dev/null)
        [ "$code" = "200" ] && break
        kill -0 $pid 2>/dev/null || { echo "ARM DIED ($1)"; return 1; }
        sleep 2
    done
    [ "$code" = "200" ] || { echo "ARM TIMEOUT ($1)"; kill $pid 2>/dev/null; return 1; }
    curl -s http://127.0.0.1:8233/v1/chat/completions -H "Content-Type: application/json" \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":300,\"temperature\":0,\"seed\":7}" \
      | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" > "$2"
    kill $pid 2>/dev/null; sleep 3
    while ss -tln 2>/dev/null | grep -q ':8233 '; do sleep 1; done
    return 0
}
run_arm /home/erol/ai/turboquant/turboquant-sync/build/bin/llama-server "$OUT/eq-new.txt" || exit 1
run_arm /home/erol/ai/turboquant/turboquant-kv-cache/build-g1/bin/llama-server "$OUT/eq-prod.txt" || exit 1
if cmp -s "$OUT/eq-new.txt" "$OUT/eq-prod.txt"; then
    echo "TEXT EQUIVALENCE: IDENTICAL ($(stat -c%s "$OUT/eq-new.txt") bytes)"
else
    echo "TEXT EQUIVALENCE: DIFFER"; diff <(head -c 300 "$OUT/eq-new.txt") <(head -c 300 "$OUT/eq-prod.txt") | head -8
fi
