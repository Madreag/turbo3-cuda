# Upstream PR draft: PDL x __restrict__ race — k_get_rows_raw (+ rule note)
# (USER opens this; assistant never submits to external repos.)

## PR: "cuda: remove entry __restrict__ from k_get_rows_raw (PDL race, #24030 class)"
Branch source: cherry-pick 254b470d6 from Madreag/turbo3-cuda fix/vision-hybrid.

Body draft:
> #24030 established that PDL and `__restrict__` must be mutually exclusive:
> under programmatic stream serialization, restrict-qualified kernel-entry
> pointers license the compiler to (a) hoist loads across
> `cudaGridDependencySynchronize()` and (b) emit non-coherent (`ld.global.nc`
> / `LDG..CONSTANT`) loads that may serve stale cache lines for data the
> predecessor kernel wrote inside the overlap window.
> `k_get_rows_raw` still carries entry restricts — including on `src1_ptr`,
> the **index tensor**: a stale/hoisted index becomes a wild gather address
> (device-fault class, not just wrong data). We hit a reproducible-under-load
> device-loss pattern on SM120/WDDM that stopped with this class of fix
> plus GGML_CUDA_PDL=0.
> SASS before/after available; after the change the kernel emits fully
> coherent loads (nc=0).

Also worth raising in the #22587 PR thread (the GDN row-per-warp kernel):
its entry signature uses raw `__restrict__` on all 8 pointers while calling
ggml_cuda_pdl_sync() — same #24030 class; our merged copy device-faulted
under sustained load until converted + PDL disabled. All 10/10 loads emit
LDG..CONSTANT under restrict (SASS), and the state tensor is the tightest
same-address producer-consumer chain in hybrid graphs.
