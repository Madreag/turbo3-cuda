#include "turbo-sink.cuh"
#include "turbo-quant.cuh"

#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <mutex>

// ─── Environment variable ─────────────────────────────────────────────────────

int turbo_sink_size() {
    static int cached = -1;
    if (cached < 0) {
        const char * env = getenv("TURBO_SINK_SIZE");
        cached = env ? atoi(env) : 0;
        if (cached > 0) {
            fprintf(stderr, "[turbo] attention sinks: first %d positions at fp16 precision\n", cached);
        }
    }
    return cached;
}

// ─── Host-side buffer management ──────────────────────────────────────────────

struct sink_buf_entry {
    half * buf;
    int64_t ne0;
    int     cap;
};

static std::mutex g_sink_mutex;
static std::unordered_map<void *, sink_buf_entry> g_sink_K_bufs;
static std::unordered_map<void *, sink_buf_entry> g_sink_V_bufs;

static half * sink_get_or_alloc(std::unordered_map<void *, sink_buf_entry> & map,
                                void * key, int64_t ne0, int sink_size) {
    std::lock_guard<std::mutex> lock(g_sink_mutex);
    auto it = map.find(key);
    if (it != map.end() && it->second.ne0 == ne0 && it->second.cap >= sink_size) {
        return it->second.buf;
    }
    if (it != map.end() && it->second.buf) {
        cudaFree(it->second.buf);
    }
    half * buf = nullptr;
    cudaMalloc(&buf, ne0 * sink_size * sizeof(half));
    cudaMemset(buf, 0, ne0 * sink_size * sizeof(half));
    map[key] = {buf, ne0, sink_size};
    return buf;
}

half * turbo_sink_get_buf(void * tensor_data, int64_t ne0) {
    const int ss = turbo_sink_size();
    if (ss <= 0) return nullptr;
    return sink_get_or_alloc(g_sink_K_bufs, tensor_data, ne0, ss);
}

half * turbo_sink_get_V_buf(void * tensor_data, int64_t ne0) {
    const int ss = turbo_sink_size();
    if (ss <= 0) return nullptr;
    return sink_get_or_alloc(g_sink_V_bufs, tensor_data, ne0, ss);
}

// ─── Capture kernel: WHT-rotate + store fp16 for sink positions ───────────────

template <typename idx_t, int GROUP_SIZE>
static __global__ void k_turbo_sink_capture(
    const float * __restrict__ src0,
    const idx_t * __restrict__ src1,
    half * __restrict__ sink_buf,
    const int64_t ne00,
    const int64_t ne01,
    const int64_t ne11,
    const int64_t s01,
    const int64_t s02,
    const int64_t s03,
    const int64_t ne12,
    const int64_t s10,
    const int64_t s11,
    const int64_t s12,
    const int64_t nb1,
    const int64_t nb2,
    const int64_t nb3,
    const int sink_size) {

    const int j = threadIdx.x;
    const int64_t n_groups_per_row = ne00 / GROUP_SIZE;
    const int64_t g = blockIdx.x;
    const int64_t i_grp = g % n_groups_per_row;
    int64_t       tmp   = g / n_groups_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne12;
    const int64_t i03   = tmp / ne12;

    const int64_t i10 = i01;
    const int64_t i11 = i01 % ne11;
    const int64_t i12 = i02;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    if (dst_row >= sink_size) return;

    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;

    __shared__ float x[GROUP_SIZE];
    x[j] = src_row[i_grp * GROUP_SIZE + j];
    __syncthreads();

    if (d_innerq_active) {
        x[j] *= d_innerq_scale[j];
    }
    __syncthreads();

    // L2 norm
    constexpr int n_warps = GROUP_SIZE / 32;
    __shared__ float warp_accum[n_warps];
    float v2 = x[j] * x[j];
    for (int offset = 16; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset);
    if (j % 32 == 0) warp_accum[j / 32] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm = sqrtf(s_norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    x[j] *= inv_norm;
    __syncthreads();

    // Forward WHT
    if (GROUP_SIZE == 128) { x[j] *= TURBO_WHT_SIGNS1[j]; }
    else                   { x[j] *= TURBO_WHT_SIGNS1_64[j]; }
    __syncthreads();

#define WHT_STAGE(h) \
    if (j % (2*(h)) < (h)) { float a = x[j], b = x[j+(h)]; x[j] = a+b; x[j+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE(1) WHT_STAGE(2) WHT_STAGE(4) WHT_STAGE(8)
    WHT_STAGE(16) WHT_STAGE(32)
    if (GROUP_SIZE == 128) { WHT_STAGE(64) }
#undef WHT_STAGE

    constexpr float inv_sqrt_group = (GROUP_SIZE == 128) ? 0.08838834764831845f : 0.125f;
    if (GROUP_SIZE == 128) { x[j] = x[j] * inv_sqrt_group * TURBO_WHT_SIGNS2[j]; }
    else                   { x[j] = x[j] * inv_sqrt_group * TURBO_WHT_SIGNS2_64[j]; }
    __syncthreads();

    // Store WHT-rotated value at full fp16 precision
    const int64_t global_col = i_grp * GROUP_SIZE + j;
    sink_buf[dst_row * ne00 + global_col] = __float2half(x[j] * grp_norm);
}

template<typename idx_t>
void turbo_sink_capture_turbo3_impl(
    const float * src0,
    const idx_t * src1,
    void * dst_data,
    int64_t ne00, int64_t ne01, int64_t ne11,
    int64_t s01, int64_t s02, int64_t s03,
    int64_t ne12, int64_t ne13,
    int64_t s10, int64_t s11, int64_t s12,
    int64_t nb1, int64_t nb2, int64_t nb3,
    int group_size,
    cudaStream_t stream) {

    const int ss = turbo_sink_size();
    if (ss <= 0) return;

    half * buf = turbo_sink_get_buf(dst_data, ne00);
    if (!buf) return;

    const int64_t n_groups_per_row = ne00 / group_size;
    const int64_t ne_total = n_groups_per_row * ne01 * ne12 * ne13;

    if (group_size == 128) {
        k_turbo_sink_capture<idx_t, 128><<<(int)ne_total, 128, 0, stream>>>(
            src0, src1, buf, ne00, ne01, ne11,
            s01, s02, s03, ne12, s10, s11, s12,
            nb1, nb2, nb3, ss);
    } else {
        k_turbo_sink_capture<idx_t, 64><<<(int)ne_total, 64, 0, stream>>>(
            src0, src1, buf, ne00, ne01, ne11,
            s01, s02, s03, ne12, s10, s11, s12,
            nb1, nb2, nb3, ss);
    }
}

// Explicit template instantiations
template void turbo_sink_capture_turbo3_impl<int32_t>(const float*, const int32_t*, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, cudaStream_t);
template void turbo_sink_capture_turbo3_impl<int64_t>(const float*, const int64_t*, void*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int, cudaStream_t);

// ─── Not used yet — device state is per-TU in fattn-common.cuh ───────────────

void turbo_sink_set_device_state(
    const half * K_buf, const half * V_buf,
    int sink_size, int64_t ne0,
    cudaStream_t stream) {
    // Device state is set directly in fattn-common.cuh via static __device__ variables.
    // This function is a placeholder for future graph-compatible implementation.
    (void)K_buf; (void)V_buf; (void)sink_size; (void)ne0; (void)stream;
}
