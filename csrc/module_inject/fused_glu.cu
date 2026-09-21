// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

// Native CUDA kernel for the segment-KI fused_glu op: computes
// out = silu(hidden[:, :k]) * hidden[:, k:2k] on the fused gate|up GEMM
// output without materializing chunked views. hidden is contiguous
// [N, 2k]; out is contiguous [N, k]. Activation math runs in fp32 with a
// single rounding to the storage dtype (torch opmath convention).

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define SILU(x) ((x) / (1.0f + expf(-(x))))

__global__ void silu_mul_halves_fp32(const float* __restrict__ hidden,
                                     float* __restrict__ out,
                                     int64_t total,
                                     int64_t k)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int64_t row = i / k;
        int64_t col = i - row * k;
        int64_t base = row * (2 * k);
        out[i] = SILU(hidden[base + col]) * hidden[base + k + col];
    }
}

__global__ void silu_mul_halves_bf16(const __nv_bfloat16* __restrict__ hidden,
                                     __nv_bfloat16* __restrict__ out,
                                     int64_t total,
                                     int64_t k)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int64_t row = i / k;
        int64_t col = i - row * k;
        int64_t base = row * (2 * k);
        float g = __bfloat162float(hidden[base + col]);
        float u = __bfloat162float(hidden[base + k + col]);
        out[i] = __float2bfloat16(SILU(g) * u);
    }
}

at::Tensor fused_silu_mul_halves(at::Tensor hidden)
{
    TORCH_CHECK(hidden.is_cuda(), "fused_silu_mul_halves is CUDA-only");
    TORCH_CHECK(hidden.dim() >= 1 && hidden.size(-1) % 2 == 0,
                "last dim must be even (gate|up layout)");
    TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
    auto sizes = hidden.sizes().vec();
    sizes.back() /= 2;
    auto out = at::empty(sizes, hidden.options());
    int64_t k = sizes.back();
    int64_t total = out.numel();
    if (total == 0) return out;

    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    TORCH_CHECK(blocks <= INT32_MAX, "fused_silu_mul_halves: tensor too large");

    if (hidden.scalar_type() == at::ScalarType::Float) {
        silu_mul_halves_fp32<<<blocks, threads, 0, stream>>>(
            hidden.data_ptr<float>(), out.data_ptr<float>(), total, k);
    } else if (hidden.scalar_type() == at::ScalarType::BFloat16) {
        silu_mul_halves_bf16<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
            total,
            k);
    } else {
        TORCH_CHECK(false, "fused_silu_mul_halves supports float32 and bfloat16");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// GDN gating for the segment-KI fused_gdn op: per-token, per-value-head
//   beta = sigmoid(b);  g = -exp(A_log) * softplus(a + dt_bias)
// computed in fp32 regardless of storage dtype (matches the native forward's
// .float() upcast, which keeps A from reaching -inf in fp16/bf16).
__global__ void gdn_gates_kernel(const __nv_bfloat16* __restrict__ a,
                                 const __nv_bfloat16* __restrict__ b,
                                 const float* __restrict__ a_log,
                                 const float* __restrict__ dt_bias,
                                 __nv_bfloat16* __restrict__ beta_out,
                                 __nv_bfloat16* __restrict__ g_out,
                                 int64_t total,
                                 int num_heads,
                                 int64_t row_stride)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        int64_t tok = i / num_heads;
        int h = (int)(i - tok * num_heads);
        // a/b arrive as [tokens, num_heads] slices of the fused GEMM output,
        // contiguous within each row but with a fused-width row stride.
        int64_t off = tok * row_stride + h;
        float av = __bfloat162float(a[off]);
        float bv = __bfloat162float(b[off]);
        float sp_x = av + dt_bias[h];
        // Numerically stable softplus: for large x, log(1+exp(x)) == x to fp32
        // precision and the naive form loses significant digits.
        float sp = (sp_x > 20.0f) ? sp_x : logf(1.0f + expf(sp_x));
        beta_out[i] = __float2bfloat16(1.0f / (1.0f + expf(-bv)));
        g_out[i] = __float2bfloat16(-expf(a_log[h]) * sp);
    }
}

std::vector<at::Tensor> gdn_gates(at::Tensor a, at::Tensor b, at::Tensor a_log, at::Tensor dt_bias)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "gdn_gates is CUDA-only");
    TORCH_CHECK(a.stride(-1) == 1 && b.stride(-1) == 1, "a/b must be unit-stride in the last dim");
    auto beta = at::empty_like(a);
    auto g = at::empty_like(b);
    int64_t total = a.numel();
    if (total == 0) return {beta, g};
    int heads = (int)dt_bias.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    gdn_gates_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr<at::BFloat16>()),
        a_log.data_ptr<float>(),
        dt_bias.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(beta.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(g.data_ptr<at::BFloat16>()),
        total,
        heads,
        (int64_t)a.stride(-2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {beta, g};
}

// Fused decode step: argmax(logits) -> write token -> advance write_pos ->
// reveal mask -> record token in output buffer. Eliminates 6 Python->CUDA
// dispatches per decode step into one kernel call.
__global__ void decode_step_kernel(const __nv_bfloat16* __restrict__ logits,
                                   int64_t* __restrict__ token_out,
                                   int64_t* __restrict__ write_pos,
                                   bool* __restrict__ mask,
                                   int64_t* __restrict__ out_buf,
                                   int step,
                                   int vocab_size,
                                   int max_len)
{
    __shared__ int s_idx[1024];
    __shared__ float s_val[1024];

    int tid = threadIdx.x;
    int local_idx = 0;
    float local_val = -INFINITY;

    for (int v = tid; v < vocab_size; v += blockDim.x) {
        float val = __bfloat162float(logits[v]);
        if (val > local_val) {
            local_val = val;
            local_idx = v;
        }
    }
    s_idx[tid] = local_idx;
    s_val[tid] = local_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int64_t best = (int64_t)s_idx[0];
        token_out[0] = best;
        out_buf[step] = best;
        int64_t new_pos = write_pos[0] + 1;
        write_pos[0] = new_pos;
        if (new_pos + 1 < max_len) { mask[new_pos + 1] = true; }
    }
}

void decode_step(at::Tensor logits,
                 at::Tensor token_out,
                 at::Tensor write_pos,
                 at::Tensor mask,
                 at::Tensor out_buf,
                 int64_t step)
{
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::ScalarType::BFloat16,
                "logits must be CUDA bf16");
    TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
    int vocab = (int)logits.numel();
    int max_len = (int)mask.size(-1);
    auto stream = at::cuda::getCurrentCUDAStream();
    decode_step_kernel<<<1, 1024, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        token_out.data_ptr<int64_t>(),
        write_pos.data_ptr<int64_t>(),
        mask.data_ptr<bool>(),
        out_buf.data_ptr<int64_t>(),
        (int)step,
        vocab,
        max_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Graph-capturable variant: no host-side step parameter. The token output
// index is derived from the GPU-resident write_pos, which the kernel itself
// advances, making the whole step self-contained for CUDA graph replay.
__global__ void decode_step_graph_kernel(const __nv_bfloat16* __restrict__ logits,
                                         int64_t* __restrict__ token_out,
                                         int64_t* __restrict__ write_pos,
                                         bool* __restrict__ mask,
                                         int64_t* __restrict__ out_buf,
                                         int vocab_size,
                                         int max_len)
{
    __shared__ int s_idx[1024];
    __shared__ float s_val[1024];

    int tid = threadIdx.x;
    int local_idx = 0;
    float local_val = -INFINITY;

    for (int v = tid; v < vocab_size; v += blockDim.x) {
        float val = __bfloat162float(logits[v]);
        if (val > local_val) {
            local_val = val;
            local_idx = v;
        }
    }
    s_idx[tid] = local_idx;
    s_val[tid] = local_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int64_t best = (int64_t)s_idx[0];
        token_out[0] = best;
        // The output token becomes the NEXT step's input; write it one
        // position ahead so it does not overwrite the current position
        // (which holds the token being processed this step).
        int64_t new_pos = write_pos[0] + 1;
        write_pos[0] = new_pos;
        out_buf[new_pos] = best;
        if (new_pos + 1 < max_len) { mask[new_pos + 1] = true; }
    }
}

void decode_step_graph(at::Tensor logits,
                       at::Tensor token_out,
                       at::Tensor write_pos,
                       at::Tensor mask,
                       at::Tensor out_buf)
{
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == at::ScalarType::BFloat16,
                "logits must be CUDA bf16");
    int vocab = (int)logits.numel();
    int max_len = (int)mask.size(-1);
    auto stream = at::cuda::getCurrentCUDAStream();
    decode_step_graph_kernel<<<1, 1024, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
        token_out.data_ptr<int64_t>(),
        write_pos.data_ptr<int64_t>(),
        mask.data_ptr<bool>(),
        out_buf.data_ptr<int64_t>(),
        vocab,
        max_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ─── Plan B: dual-weight GEMV (b=1 decode) ───
// Reads gate.weight and up.weight directly — no concat, no weight copies.
// Each warp handles one output feature: lanes cooperatively compute two
// dot products (gate row and up row share the same hidden vector read),
// then applies silu(a)*b.  Memory-bound at b=1; coalesced reads achieve
// near-peak HBM bandwidth with zero tiling or tensor-core complexity.

__global__ void dual_gemv_silu_mul_kernel(const __nv_bfloat16* __restrict__ hidden,
                                          const __nv_bfloat16* __restrict__ gate_w,
                                          const __nv_bfloat16* __restrict__ up_w,
                                          __nv_bfloat16* __restrict__ out,
                                          int out_features,
                                          int in_features)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= out_features) return;

    const __nv_bfloat16* gate_row = gate_w + (long)warp_id * in_features;
    const __nv_bfloat16* up_row = up_w + (long)warp_id * in_features;

    // Each lane accumulates every 32nd element (coalesced across lanes).
    float gate_acc = 0.0f, up_acc = 0.0f;
    for (int i = lane; i < in_features; i += 32) {
        float h = __bfloat162float(hidden[i]);
        gate_acc += h * __bfloat162float(gate_row[i]);
        up_acc += h * __bfloat162float(up_row[i]);
    }

    // Warp-level reduction for both dot products simultaneously.
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_acc += __shfl_down_sync(0xffffffff, gate_acc, offset);
        up_acc += __shfl_down_sync(0xffffffff, up_acc, offset);
    }

    if (lane == 0) {
        float activated = SILU(gate_acc);
        out[warp_id] = __float2bfloat16_rn(activated * up_acc);
    }
}

void dual_gemv_silu_mul(at::Tensor hidden, at::Tensor gate_w, at::Tensor up_w, at::Tensor out)
{
    TORCH_CHECK(hidden.is_cuda() && hidden.scalar_type() == at::ScalarType::BFloat16,
                "hidden must be CUDA bf16");
    TORCH_CHECK(gate_w.is_contiguous() && up_w.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(gate_w.size(0) == up_w.size(0) && gate_w.size(1) == up_w.size(1),
                "gate/up weight shapes must match");
    int out_f = (int)gate_w.size(0);
    int in_f = (int)gate_w.size(1);
    int warps_per_block = 8;  // 256 threads
    int blocks = (out_f + warps_per_block - 1) / warps_per_block;
    auto stream = at::cuda::getCurrentCUDAStream();
    dual_gemv_silu_mul_kernel<<<blocks, warps_per_block * 32, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(hidden.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(gate_w.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(up_w.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        out_f,
        in_f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ─── Custom b=1 decode attention (graph-compatible, no mask) ───
// Streams softmax(q @ K[0:pos]^T / sqrt(d)) @ V[0:pos] with GQA. The valid
// KV length comes from a GPU-resident write_pos tensor, so the kernel is
// CUDA-graph replayable (address fixed, only the value changes per step).
// One block per KV head; its 8 warps split as (query head, KV chunk):
// warp w handles query head w % q_per_kv over KV positions strided by
// split = 8 / q_per_kv, then per-head partial online-softmax states are
// merged through shared memory. Each lane owns a contiguous 16B slice of
// the head dimension (HEAD_DIM/32 elements) for coalesced K/V row loads.

template <int HEAD_DIM>
__global__ void __launch_bounds__(512) decode_attn_kernel(const __nv_bfloat16* __restrict__ q,
                                                          const __nv_bfloat16* __restrict__ K,
                                                          const __nv_bfloat16* __restrict__ V,
                                                          const int64_t* __restrict__ write_pos,
                                                          __nv_bfloat16* __restrict__ out,
                                                          int num_q_heads,
                                                          int num_kv_heads,
                                                          int max_len,
                                                          float scale)
{
    constexpr int EPL = HEAD_DIM / 32;  // head-dim elements owned per lane
    const int q_per_kv = num_q_heads / num_kv_heads;
    const int warps_per_block = blockDim.x >> 5;
    const int split = warps_per_block;  // KV chunks per query head

    // One block per query head: all the block's warps split its KV range,
    // giving enough independent streams to cover HBM latency at b=1.
    const int q_head = blockIdx.x;
    const int kv_head = q_head / q_per_kv;
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int s = warp_id;

    const int pos = (int)(*write_pos);  // valid KV positions: 0..pos inclusive

    const __nv_bfloat16* Kg = K + (long)kv_head * max_len * HEAD_DIM;
    const __nv_bfloat16* Vg = V + (long)kv_head * max_len * HEAD_DIM;

    float q_reg[EPL];
#pragma unroll
    for (int e = 0; e < EPL; e++)
        q_reg[e] = __bfloat162float(q[(long)q_head * HEAD_DIM + lane * EPL + e]);

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float out_reg[EPL];
#pragma unroll
    for (int e = 0; e < EPL; e++) out_reg[e] = 0.0f;

    // UNROLL positions in flight so their K loads overlap and hide HBM latency.
    constexpr int UNROLL = 4;
    for (int base = s; base <= pos; base += UNROLL * split) {
        float k_reg[UNROLL][EPL];
        int idx[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            idx[u] = base + u * split;
            if (idx[u] <= pos) {
                const __nv_bfloat16* krow = Kg + (long)idx[u] * HEAD_DIM;
#pragma unroll
                for (int e = 0; e < EPL; e++) k_reg[u][e] = __bfloat162float(krow[lane * EPL + e]);
            }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
            if (idx[u] > pos) continue;  // uniform across the warp
            float dot = 0.0f;
#pragma unroll
            for (int e = 0; e < EPL; e++) dot += q_reg[e] * k_reg[u][e];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffff, dot, off);
            float score = __shfl_sync(0xffffffff, dot, 0) * scale;

            float new_max = fmaxf(max_score, score);
            float rescale = expf(max_score - new_max);  // 0 when max_score is still -inf
            float weight = expf(score - new_max);
            sum_exp = sum_exp * rescale + weight;
#pragma unroll
            for (int e = 0; e < EPL; e++) out_reg[e] *= rescale;
            const __nv_bfloat16* vrow = Vg + (long)idx[u] * HEAD_DIM;
#pragma unroll
            for (int e = 0; e < EPL; e++)
                out_reg[e] += weight * __bfloat162float(vrow[lane * EPL + e]);
            max_score = new_max;
        }
    }

    // Merge the per-chunk partial softmax states (max, sum, weighted V) in
    // shared memory; an empty chunk contributes 0 via exp(-inf - M).
    __shared__ float s_max[16];
    __shared__ float s_sum[16];
    __shared__ float s_out[16][HEAD_DIM];

    s_max[warp_id] = max_score;
    s_sum[warp_id] = sum_exp;
#pragma unroll
    for (int e = 0; e < EPL; e++) s_out[warp_id][lane * EPL + e] = out_reg[e];
    __syncthreads();

    if (s == 0) {  // the first chunk's warp combines all chunks for its head
        float M = -INFINITY;
        for (int t = 0; t < split; t++) M = fmaxf(M, s_max[t]);
        float S = 0.0f;
        float acc[EPL];
#pragma unroll
        for (int e = 0; e < EPL; e++) acc[e] = 0.0f;
        for (int t = 0; t < split; t++) {
            float wm = expf(s_max[t] - M);
            S += s_sum[t] * wm;
#pragma unroll
            for (int e = 0; e < EPL; e++) acc[e] += s_out[t][lane * EPL + e] * wm;
        }
        float inv = 1.0f / S;
#pragma unroll
        for (int e = 0; e < EPL; e++)
            out[(long)q_head * HEAD_DIM + lane * EPL + e] = __float2bfloat16_rn(acc[e] * inv);
    }
}

void decode_attn(at::Tensor q,
                 at::Tensor K,
                 at::Tensor V,
                 at::Tensor write_pos,
                 at::Tensor out,
                 int64_t num_q_heads,
                 int64_t num_kv_heads,
                 int64_t head_dim,
                 int64_t max_len)
{
    TORCH_CHECK(q.is_cuda() && q.scalar_type() == at::ScalarType::BFloat16, "q must be CUDA bf16");
    TORCH_CHECK(K.is_contiguous() && V.is_contiguous(), "K/V must be contiguous");
    TORCH_CHECK(write_pos.is_cuda() && write_pos.scalar_type() == at::ScalarType::Long,
                "write_pos must be a CUDA int64 tensor");
    TORCH_CHECK(num_q_heads % num_kv_heads == 0,
                "GQA requires num_q_heads divisible by num_kv_heads");
    int q_per_kv = (int)(num_q_heads / num_kv_heads);
    TORCH_CHECK(head_dim % 32 == 0, "head_dim must be a multiple of 32, got ", head_dim);
    // 16 warps/block (512 threads) so each query head gets a 4-way KV split
    // — the per-warp stream is otherwise latency-bound at these sizes. More
    // warps than this exhaust registers with __launch_bounds__(512).
    const int warps_per_block = 16;
    TORCH_CHECK(warps_per_block % q_per_kv == 0,
                "q heads per KV head (",
                q_per_kv,
                ") must divide ",
                warps_per_block);

    float scale = 1.0f / sqrtf((float)head_dim);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto q_ptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
    auto k_ptr = reinterpret_cast<const __nv_bfloat16*>(K.data_ptr<at::BFloat16>());
    auto v_ptr = reinterpret_cast<const __nv_bfloat16*>(V.data_ptr<at::BFloat16>());
    auto o_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());

    switch (head_dim) {
        case 64:
            decode_attn_kernel<64>
                <<<num_q_heads, warps_per_block * 32, 0, stream>>>(q_ptr,
                                                                   k_ptr,
                                                                   v_ptr,
                                                                   write_pos.data_ptr<int64_t>(),
                                                                   o_ptr,
                                                                   (int)num_q_heads,
                                                                   (int)num_kv_heads,
                                                                   (int)max_len,
                                                                   scale);
            break;
        case 128:
            decode_attn_kernel<128>
                <<<num_q_heads, warps_per_block * 32, 0, stream>>>(q_ptr,
                                                                   k_ptr,
                                                                   v_ptr,
                                                                   write_pos.data_ptr<int64_t>(),
                                                                   o_ptr,
                                                                   (int)num_q_heads,
                                                                   (int)num_kv_heads,
                                                                   (int)max_len,
                                                                   scale);
            break;
        case 256:
            decode_attn_kernel<256>
                <<<num_q_heads, warps_per_block * 32, 0, stream>>>(q_ptr,
                                                                   k_ptr,
                                                                   v_ptr,
                                                                   write_pos.data_ptr<int64_t>(),
                                                                   o_ptr,
                                                                   (int)num_q_heads,
                                                                   (int)num_kv_heads,
                                                                   (int)max_len,
                                                                   scale);
            break;
        default: TORCH_CHECK(false, "Unsupported head_dim: ", head_dim);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("decode_attn", &decode_attn, "b=1 decode attention with GQA, graph-compatible (CUDA)");
    m.def("decode_step", &decode_step, "fused decode step update (CUDA)");
    m.def("decode_step_graph", &decode_step_graph, "graph-capturable decode step (CUDA)");
    m.def("gdn_gates", &gdn_gates, "fused GDN beta/g gating (CUDA)");
    m.def("fused_silu_mul_halves",
          &fused_silu_mul_halves,
          "fused silu(gate)*up on [N, 2k] hidden (CUDA)");
    m.def("dual_gemv_silu_mul",
          &dual_gemv_silu_mul,
          "b=1 GEMV reading gate/up weights separately: silu(h*Wg)*(h*Wu) (CUDA)");
}
