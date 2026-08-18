# Qwen3.6-era discriminator — 2026-08-18 (user-directed)
Q: does the pre-3.8 known-good stack (3.6-era fork binaries + Qwen3.6) crash this box?
Stack: turboquant-g1/build/bin (Aug 14 12:56, build 8815, last pre-sync 3.6-era build,
matched libs, era llama-perplexity). Model: HF Qwen3.6-27B Q6_K (prod file was deleted;
same arch/format class). Same driver 610.47 as both the stable month and every crash.
Phase A: era llama-perplexity + calib_v2.txt -c 4096 (same corpus/chunking that killed
  official b10488 + Qwen3.8 at ~2 min). PASS = full ~385-chunk completion.
Phase B: era server, frozen start.sh 3.6 config, continuous load, hours.
DEATH at any point => machine-level cause (driver-state or hardware), NOT 3.8-specific.
FULL SURVIVAL => trigger is 3.8-class workload x driver/hw; levers: DDU+retest, power-cap
  retest, or ship 3.6/3090 for vacation.
milestones.log + telemetry.csv in this dir are written continuously for post-mortem.
