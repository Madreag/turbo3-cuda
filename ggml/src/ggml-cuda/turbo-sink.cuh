#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

int turbo_sink_size();

half * turbo_sink_get_buf(void * tensor_data, int64_t ne0);
half * turbo_sink_get_V_buf(void * tensor_data, int64_t ne0);
half * turbo_sink_lookup_buf(void * tensor_data, int64_t * out_ne0);

template<typename idx_t>
void turbo_sink_capture_turbo3_impl(
    const float * src0, const idx_t * src1, void * dst_data,
    int64_t ne00, int64_t ne01, int64_t ne11,
    int64_t s01, int64_t s02, int64_t s03,
    int64_t ne12, int64_t ne13,
    int64_t s10, int64_t s11, int64_t s12,
    int64_t nb1, int64_t nb2, int64_t nb3,
    int group_size, cudaStream_t stream);

void turbo_sink_set_device_state(
    const half * K_buf, const half * V_buf,
    int sink_size, int64_t ne0,
    cudaStream_t stream);
