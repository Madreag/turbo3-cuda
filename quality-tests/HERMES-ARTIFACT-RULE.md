# Hermes artifact-mode rule (R3 deliverable — ready to paste, NOT yet deployed)

Add to Hermes's system prompt (artifact/creative-web section):

---
WHEN ASKED FOR A SINGLE-FILE VISUAL/3D HTML ARTIFACT (scenes, demos, generative art):
1. Start from the voxel chassis template verbatim (three.js importmap, error overlay,
   renderer + OrbitControls + lights, `addVoxel(x,y,z,color,sx,sy,sz)`, `section(name,fn)`,
   render loop started BEFORE scene build). Do not modify the chassis.
2. Write ONLY the SCENE section: `section('name', () => { ... })` blocks composed of
   addVoxel calls and simple bounded loops (`for (let i = 0; i < N; i++)` with literal
   integer N). No new helpers, no while-loops, no recursion, no external assets.
3. Deliver the complete file in ONE write_file call.
---

Chassis source of truth: `quality-tests/voxel-scaffold.html` (branch hermes/server-foundation),
render-proven standalone (brightness 33, 6 hues, 0 console errors, headless Chrome).

R3 deploy procedure (on "go"): hand this rule + chassis to the Hermes agent for its
system prompt → one Hermes-shaped run through the proxy → wire-captured artifact →
headless render must pass pixels (brightness >25, ≥8 hues, 0 uncaught).
R2 precedes it: 5/5 scaffold-fill battery via HERMES_RECIPE=2 harness mode (to be added:
prompt = chassis + "fill the SCENE section only").
