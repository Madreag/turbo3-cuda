# 3090 FALLBACK SERVING — deploy in ~20 minutes

Vacation-safety package (2026-08-18): serves Qwen3.8-27B on the RTX 3090 24GB box
with the same advertised model id (`qwen3.8-27b-320k`), so Hermes needs **zero
config changes** beyond pointing at the new box's IP.

**What you get:** ~40-60 t/s decode (community-proven turboquant-on-3090 class),
96K real context, MTP n3 speculative, turbo4 KV, vision on CPU. Weights are our
own Q4_K_M quantized with the workload-calibrated imatrix v2.
**Honest label:** this quant is QUALITY-UNGATED (no KLD/battery — gates need a
stable GPU). It is calibration-matched but unproven; acceptable for a fallback.

## Prerequisites on the 3090 box
- NVIDIA driver installed (any 2025+ version; it includes WSL CUDA support).
- WSL2 + Ubuntu (`wsl --install -d Ubuntu` in admin PowerShell if absent, then reboot).
- `python3` inside WSL (`sudo apt install -y python3` if missing) — for the proxy.

## Deploy steps
1. Copy the staged package folder `D:\spill\fallback-3090\` from the 5090 box to
   the 3090 box (LAN share / USB — ~17 GB, mostly the model). Put it anywhere
   inside WSL, e.g. `~/fallback-3090/`. IMPORTANT: copy INTO the WSL filesystem
   (`\\wsl.localhost\Ubuntu\home\<user>\`), not a /mnt/c path — model load speed.
2. Copy `api.key` from the 5090 box (`~/.config/llama-tcq/api.key`) into the
   package folder. (Keys never live in git — this file ships separately.)
3. `chmod +x ~/fallback-3090/*.sh`
4. `bash ~/fallback-3090/start-3090.sh` — first load is cold-disk slow; the
   launcher waits up to 8 min. Exit 0 + the two PID lines = UP.
5. `bash ~/fallback-3090/status-3090.sh` — expect health 200 and ~22.5-23 GB used.
6. Smoke test from any LAN machine:
   `curl http://<3090-box-ip>:8130/v1/models -H "Authorization: Bearer <key>"`
7. Point Hermes at `http://<3090-box-ip>:8130/v1`. Model id stays
   `qwen3.8-27b-320k`.
8. Windows firewall: first run may prompt — allow, or pre-open port 8130
   (admin PowerShell):
   `New-NetFirewallRule -DisplayName tcq8130 -Direction Inbound -LocalPort 8130 -Protocol TCP -Action Allow`
   WSL port reachability from LAN needs the usual netsh portproxy if WSL is NAT'd:
   `netsh interface portproxy add v4tov4 listenport=8130 listenaddress=0.0.0.0 connectport=8130 connectaddress=(wsl hostname -I)`

## Operational notes
- REAL CTX IS 96K under the 320k alias (VRAM truth on 24 GB). Keep vacation
  sessions under ~90K tokens; typical Hermes sessions run well below this.
- Stretch profile: `CTX=131072 bash start-3090.sh` fits ONLY if the box is
  headless (no monitor eating VRAM) — ~0.7 GB slack, not the default.
- `stop-3090.sh` is the only sanctioned stop. Detach law is inherited: closing
  terminals/sessions must not kill the stack (setsid + nohup + </dev/null).
- Slot files in `slots-3090/` are config-specific — delete them if you change
  CTX or KV types.
- Vision: mmproj runs on CPU (`--no-mmproj-offload`) — image encode is slow
  (~30-90s on a weaker CPU), text speed unaffected.

## Package contents (staged at D:\spill\fallback-3090\)
- `bin/` — sm_86 build of the prod fork source (branch fix/vision-hybrid
  @ 007892f31, same code as 5090 prod) + libs
- `models/qwen38-q4km-imx2.gguf` — Q4_K_M, imatrix-v2 calibrated (~16.5 GB)
- `models/mmproj-F16.gguf` — gate-validated original vision projector
- `proxy.py`, `start-3090.sh`, `stop-3090.sh`, `status-3090.sh`, this README
- NOT included: `api.key` (copy separately, never in git)

## Validation status (honest)
- Model file, tokenizer/template, alias, and server boot: smoke-tested on the
  build box (CPU-only config — the turbo4+FA+MTP graph needs CUDA, so the FULL
  prod-config path is first exercised on the 3090 itself; same source already
  runs this exact config in prod on the 5090).
- If the 3090 first boot errors on the KV path, fallback flags that always
  work: replace the two `-ct*` lines with `-ctk q8_0 -ctv q8_0` and drop the
  two `--spec-*` args (costs ctx headroom + speed, keeps you serving).

## VRAM budget (why 96K default)
Q4_K_M weights ~16.5 GB + KV turbo4 @96K ~1.7 GB + recurrent states/copies
~2.5 GB + compute buffers ~1.8 GB ≈ 22.5 GB of 24 GB → ~1.3 GB slack.
At 131K: ~23.1 GB → headless-only.
