#include "common.cuh"
#include "lightning-indexer.cuh"
#include "fattn-common.cuh"
#include "convert.cuh"

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
#if defined(TURING_MMA_AVAILABLE)

typedef union {
    int2 i2;
    half2 h2[2];
} half4;

// TODO add support for AMD cards via rocWMMA
#include <mma.h>
namespace wmma = nvcuda::wmma;

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int HEADS_PER_INNER_LOOP = 8;
    constexpr int K_EMBD_PER_INNER_LOOP = 16;
    constexpr int N_EMBD_PADDED = N_EMBD + 8;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each block processes K_VECS_PER_BLOCK K vectors
    const int start_kv = blockIdx.x * K_VECS_PER_BLOCK;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load weights and first Q tile to shared memory

    __shared__ float w_shared[N_HEAD];
    __shared__ int2  q_shared_h[HEADS_PER_INNER_LOOP][N_EMBD_PADDED / 4];

    if (tid < N_HEAD) {
        w_shared[tid] = w_base[tid];
    }

    // total number of half4 elements in HEADS_PER_INNER_LOOP x N_EMBD Q tile
    constexpr int N_Q_TILE = HEADS_PER_INNER_LOOP * (N_EMBD / 4);
    // number of registers needed in each thread to store Q tile in thread block
    constexpr int N_Q_NEXT = (N_Q_TILE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

#pragma unroll
    for (int i_q = tid; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
        const int i_head = i_q / (N_EMBD / 4);
        const int i_embd = i_q % (N_EMBD / 4);
        const float4 q = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        half4 q_packed;
        q_packed.h2[0] = __float22half2_rn(make_float2(q.x, q.y));
        q_packed.h2[1] = __float22half2_rn(make_float2(q.z, q.w));
        q_shared_h[i_head][i_embd] = q_packed.i2;
    }

    // phase 2 - load (and dequantize if needed) K to shared mem

    __shared__ half2 k_shared_h[K_VECS_PER_BLOCK][N_EMBD_PADDED / 4][2];

    constexpr int n_k = K_VECS_PER_BLOCK * (N_EMBD / 4);

    if constexpr (TYPE_K == GGML_TYPE_F16) {
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const int2 * k_base = (const int2 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                *(int2*) &k_shared_h[i_k_vec][i_embd] = k_base[i_embd];
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    } else {
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, half, 4>();
#pragma unroll
        for (int i_k = tid; i_k < n_k; i_k += THREADS_PER_BLOCK) {
            const int i_k_vec = i_k / (N_EMBD / 4);
            const int i_embd = i_k % (N_EMBD / 4);
            const int i_kv = start_kv + i_k_vec;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_shared_h[i_k_vec][i_embd][0], i_embd * 4);
            } else {
                *(int2*) &k_shared_h[i_k_vec][i_embd] = make_int2(0, 0);
            }
        }
    }

    __syncthreads();

    // phase 3 - calculate lightning indexer scores

    __shared__ float qk_shared[WARPS_PER_BLOCK][HEADS_PER_INNER_LOOP][K_VECS_PER_BLOCK];

    // load K fragment
    wmma::fragment<wmma::matrix_b, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::col_major> frag_k;
    wmma::load_matrix_sync(frag_k, (half*) &k_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

    float score_k = 0.0f;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += HEADS_PER_INNER_LOOP) {
        const int i_head_next = i_head_0 + HEADS_PER_INNER_LOOP;

        // we don't use accumulator for anything, fill it with zeros
        wmma::fragment<wmma::accumulator, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, float> frag_acc;
        wmma::fill_fragment(frag_acc, 0.0f);

        // load Q fragment
        wmma::fragment<wmma::matrix_a, HEADS_PER_INNER_LOOP, K_VECS_PER_BLOCK, K_EMBD_PER_INNER_LOOP, half, wmma::row_major> frag_q;
        wmma::load_matrix_sync(frag_q, (half*) &q_shared_h[0][i_warp * K_EMBD_PER_INNER_LOOP / 4], N_EMBD_PADDED);

        // preload next Q tile to registers during matrix multiplication
        float4 q_next[N_Q_NEXT];

        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_head_next + i_q / (N_EMBD / 4);
                const int i_embd =               i_q % (N_EMBD / 4);
                q_next[i_q_next++] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
            }
        }

        // perform matrix multiplication
        wmma::mma_sync(frag_acc, frag_q, frag_k, frag_acc);
        wmma::store_matrix_sync((float*) &qk_shared[i_warp][0][0], frag_acc, K_VECS_PER_BLOCK, wmma::mem_row_major);

        // make sure all threads finished using q_shared_h so we can store next tile
        __syncthreads();

        // write preloaded Q tile to shared memory
        if (i_head_next < N_HEAD) {
#pragma unroll
            for (int i_q = tid, i_q_next = 0; i_q < N_Q_TILE; i_q += THREADS_PER_BLOCK) {
                const int i_head = i_q / (N_EMBD / 4);
                const int i_embd = i_q % (N_EMBD / 4);
                half4 q_packed;
                q_packed.h2[0] = __float22half2_rn(make_float2(q_next[i_q_next].x, q_next[i_q_next].y));
                q_packed.h2[1] = __float22half2_rn(make_float2(q_next[i_q_next].z, q_next[i_q_next].w));
                q_shared_h[i_head][i_embd] = q_packed.i2;
                ++i_q_next;
            }
        }

        // accumulate QK multiplication results from all block warps
        // (there are 256 threads in block and 256 matmul outputs)
        // TODO it will break if WARP_SIZE is not 32
        const int h = tid / K_VECS_PER_BLOCK;
        const int k = tid % K_VECS_PER_BLOCK;
        const float w_val = w_shared[i_head_0 + h];

        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
            sum += qk_shared[w][h][k];
        }

        // ReLU, weight
        sum = sum > 0.0f ? sum : 0.0f;
        sum *= w_val;

        // wait until qk_shared[0] is no longer used
        __syncthreads();

        // reuse qk_shared[0] for storing partial results
        qk_shared[0][h][k] = sum;

        // wait until all threads write their results
        __syncthreads();

        // accumulate result over heads
        if (tid < K_VECS_PER_BLOCK) {
#pragma unroll
            for (int i_head = 0; i_head < HEADS_PER_INNER_LOOP; ++i_head) {
                score_k += qk_shared[0][i_head][tid];
            }
        }

        // make sure all threads finished using qk_shared
        __syncthreads();
    }

    // phase 4 - store output to VRAM

    if (tid < K_VECS_PER_BLOCK) {
        const int i_kv = start_kv + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = score_k + __half2float(m_base[i_kv]);
        }
    }
}

#else // defined(TURING_MMA_AVAILABLE)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_wmma(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {
    GGML_UNUSED_VARS(Q, K, W, M, dst,
        n_stream, n_batch, n_kv,
        nb1, nb2, nb3,
        nbq1, nbq2, nbq3,
        nbk1, nbk2, nbk3,
        nbw1, nbw2, nbw3,
        nem3);
    NO_DEVICE_CODE;
}

#endif // defined(TURING_MMA_AVAILABLE)
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// tokens scored per block by the tile kernel
#define LIGHTNING_INDEXER_TILE_TOKENS 8

// heads whose queries the tile kernel stages per pass, MUSA arch 21 caps static shared memory
// at 28 KB and the queries of four heads do not fit there next to the key tile
#if defined(GGML_USE_MUSA) && defined(__MUSA_ARCH__) && __MUSA_ARCH__ < 220
#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 2
#else
#define LIGHTNING_INDEXER_TILE_HEADS_PER_PASS 4
#endif

// TODO there is one ugly assumption used in this kernel - that WARP_SIZE is equal to 32
// thanks to that one warp operating on float4 processes whole indexer K/Q vectors
// 32 * 4 = 128 (N_EMBD)

template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_vec(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int K_VECS_PER_WARP = K_VECS_PER_BLOCK / WARPS_PER_BLOCK;
    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;

    const int i_batch  = blockIdx.y;
    const int i_stream = blockIdx.z;
    const int i_warp   = threadIdx.y;
    const int i_lane   = threadIdx.x;
    const int tid      = i_warp * WARP_SIZE + i_lane;

    // each warp processes K_VECS_PER_WARP K vectors
    const int start_kv_block = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_kv = start_kv_block + i_warp * K_VECS_PER_WARP;

    const char  * q_base = (const char  *)                 Q + i_batch*nbq2 + i_stream*nbq3;
    const float * w_base = (const float *) ((const char *) W + i_batch*nbw1 + i_stream*nbw3);

    // phase 1 - load (and dequantize if needed) K to registers

    float4 k_reg_f[K_VECS_PER_WARP];

    if constexpr (TYPE_K == GGML_TYPE_F32) {
        // direct copy of float4
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const float4 * k_base = (const float4 *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                k_reg_f[k] = k_base[i_lane];
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    } else {
        // dequantize remaining types to float
        constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            int i_kv = start_kv + k;
            if (i_kv < n_kv) {
                const void * k_base = (const void *) ((const char *) K + i_kv*nbk2 + i_stream*nbk3);
                dequantize_k(k_base, &k_reg_f[k], i_lane * 4);
            } else {
                k_reg_f[k] = make_float4(0, 0, 0, 0);
            }
        }
    }

    float score_k[K_VECS_PER_WARP] = { 0.0f };

    // load weights and Q only for N_HEAD_INNER heads at once to reduce shared memory usage
    constexpr int N_HEAD_INNER = N_HEAD / 4;

    for (int i_head_0 = 0; i_head_0 < N_HEAD; i_head_0 += N_HEAD_INNER) {
        // phase 2 - load weights and Q to shared memory

        __shared__ float  w_shared[N_HEAD_INNER];
        __shared__ float4 q_shared_f[N_HEAD_INNER][N_EMBD / 4];

        if (tid < N_HEAD_INNER) {
            w_shared[tid] = w_base[i_head_0 + tid];
        }

        constexpr int n_q = N_HEAD_INNER * (N_EMBD / 4);
#pragma unroll
        for (int i_q = tid; i_q < n_q; i_q += THREADS_PER_BLOCK) {
            const int i_head_inner = i_q / (N_EMBD / 4);
            const int i_head = i_head_0 + i_head_inner;
            const int i_embd = i_q % (N_EMBD / 4);
            q_shared_f[i_head_inner][i_embd] = *(const float4 *) (q_base + i_head*nbq1 + i_embd*sizeof(float4));
        }

        __syncthreads();

        // phase 3 - calculate lightning indexer scores

        for (int i_head_inner = 0; i_head_inner < N_HEAD_INNER; ++i_head_inner) {
            const float w_val = w_shared[i_head_inner];
            float qk[K_VECS_PER_WARP] = { 0.0f };

            // dot product of floats
            const float4 q_vec = q_shared_f[i_head_inner][i_lane];

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                ggml_cuda_mad(qk[k], q_vec.x, k_reg_f[k].x);
                ggml_cuda_mad(qk[k], q_vec.y, k_reg_f[k].y);
                ggml_cuda_mad(qk[k], q_vec.z, k_reg_f[k].z);
                ggml_cuda_mad(qk[k], q_vec.w, k_reg_f[k].w);
            }

#pragma unroll
            for (int k = 0; k < K_VECS_PER_WARP; ++k) {
                float sum = warp_reduce_sum(qk[k]);

                // ReLU, weight
                if (i_lane == 0) {
                    sum = (sum > 0.0f) ? sum : 0.0f;
                    score_k[k] += sum * w_val;
                }
            }
        }

        __syncthreads();
    }

    // phase 4 - store outputs to shared memory

    __shared__ float dst_shared[K_VECS_PER_BLOCK];

    if (i_lane == 0) {
#pragma unroll
        for (int k = 0; k < K_VECS_PER_WARP; ++k) {
            dst_shared[i_warp * K_VECS_PER_WARP + k] = score_k[k];
        }
    }

    __syncthreads();

    // phase 5 - write from shared memory to VRAM in coalesced manner

    if (tid < K_VECS_PER_BLOCK) {
        int i_kv = start_kv_block + tid;
        if (i_kv < n_kv) {
            const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
            float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);
            dst_base[i_kv] = dst_shared[tid] + __half2float(m_base[i_kv]);
        }
    }
}

// one block scores a tile of K_VECS_PER_BLOCK keys against TOKENS_PER_BLOCK tokens: the keys are
// staged in half precision and the queries of every head in float, each thread owns KEYS_PER_THREAD
// keys for one token, a warp shares its token so the query reads are broadcasts, and every key
// element is widened once for all heads, so no dot product needs a cross thread reduction
template <int WARPS_PER_BLOCK, int K_VECS_PER_BLOCK, int64_t N_EMBD, int64_t N_HEAD, ggml_type TYPE_K>
static __global__ void lightning_indexer_kernel_tile(
        const float * Q, const char * K, const float * W, const half * M, float * dst,
        int64_t n_stream, int64_t n_batch, int64_t n_kv,
        size_t nb1, size_t nb2, size_t nb3,
        size_t nbq1, size_t nbq2, size_t nbq3,
        size_t nbk1, size_t nbk2, size_t nbk3,
        size_t nbw1, size_t nbw2, size_t nbw3,
        size_t nbm1, size_t nbm2, size_t nbm3,
        int64_t nem3
    ) {

    constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;
    constexpr int TOKENS_PER_BLOCK  = LIGHTNING_INDEXER_TILE_TOKENS;
    constexpr int KEY_LANES         = THREADS_PER_BLOCK / TOKENS_PER_BLOCK;
    constexpr int KEYS_PER_THREAD   = K_VECS_PER_BLOCK / KEY_LANES;
    constexpr int N_EMBD_H2         = N_EMBD / 2;
    constexpr int HEADS_PER_PASS    = N_HEAD < LIGHTNING_INDEXER_TILE_HEADS_PER_PASS ? N_HEAD : LIGHTNING_INDEXER_TILE_HEADS_PER_PASS;

    static_assert(THREADS_PER_BLOCK % TOKENS_PER_BLOCK == 0, "threads must cover the token tile");
    static_assert(K_VECS_PER_BLOCK % KEY_LANES == 0, "key lanes must cover the key tile");
    static_assert(N_HEAD % HEADS_PER_PASS == 0, "head passes must cover the heads");

    const int tid         = threadIdx.y * WARP_SIZE + threadIdx.x;
    const int start_kv    = blockIdx.x * K_VECS_PER_BLOCK;
    const int start_batch = blockIdx.y * TOKENS_PER_BLOCK;
    const int i_stream    = blockIdx.z;

    // the row padding keeps the keys of consecutive threads in distinct banks
    __shared__ half2 k_shared[K_VECS_PER_BLOCK][N_EMBD_H2 + 1];
    __shared__ float2 q_shared[HEADS_PER_PASS][TOKENS_PER_BLOCK][N_EMBD_H2];
    __shared__ float w_shared[N_HEAD][TOKENS_PER_BLOCK];

    // phase 1 - stage the key tile four elements at a time, rows past n_kv are zero

#pragma unroll
    for (int i = tid; i < K_VECS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
        const int r  = i / (N_EMBD / 4);
        const int c4 = i % (N_EMBD / 4);

        half2 lo = make_half2(0.0f, 0.0f);
        half2 hi = lo;
        if (start_kv + r < n_kv) {
            const char * k_row = K + (start_kv + r)*nbk2 + i_stream*nbk3;
            if constexpr (TYPE_K == GGML_TYPE_F16) {
                lo = ((const half2 *) k_row)[2*c4 + 0];
                hi = ((const half2 *) k_row)[2*c4 + 1];
            } else {
                float4 v;
                if constexpr (TYPE_K == GGML_TYPE_F32) {
                    v = ((const float4 *) k_row)[c4];
                } else {
                    constexpr dequantize_V_t dequantize_k = get_dequantize_V<TYPE_K, float, 4>();
                    dequantize_k(k_row, &v, c4 * 4);
                }
                lo = make_half2(v.x, v.y);
                hi = make_half2(v.z, v.w);
            }
        }

        k_shared[r][2*c4 + 0] = lo;
        k_shared[r][2*c4 + 1] = hi;
    }

    // phase 2 - stage the weights of every head, tokens past n_batch are zero

    if (tid < N_HEAD * TOKENS_PER_BLOCK) {
        const int h = tid / TOKENS_PER_BLOCK;
        const int r = tid % TOKENS_PER_BLOCK;
        w_shared[h][r] = start_batch + r < n_batch ?
            ((const float *) ((const char *) W + (start_batch + r)*nbw1 + i_stream*nbw3))[h] : 0.0f;
    }

    const int kl = tid % KEY_LANES;
    const int tl = tid / KEY_LANES;

    float qk[N_HEAD][KEYS_PER_THREAD] = { { 0.0f } };

#pragma unroll
    for (int h0 = 0; h0 < N_HEAD; h0 += HEADS_PER_PASS) {
        // the previous pass is fully consumed before its queries are replaced
        if (h0 > 0) {
            __syncthreads();
        }

        // phase 3 - stage the queries of the heads of this pass, tokens past n_batch are zero

#pragma unroll
        for (int i = tid; i < HEADS_PER_PASS * TOKENS_PER_BLOCK * (N_EMBD / 4); i += THREADS_PER_BLOCK) {
            const int h  = i / (TOKENS_PER_BLOCK * (N_EMBD / 4));
            const int r  = i / (N_EMBD / 4) % TOKENS_PER_BLOCK;
            const int c4 = i % (N_EMBD / 4);

            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (start_batch + r < n_batch) {
                v = *(const float4 *) ((const char *) Q + (h0 + h)*nbq1 + (start_batch + r)*nbq2 + i_stream*nbq3 + c4*sizeof(float4));
            }

            q_shared[h][r][2*c4 + 0] = make_float2(v.x, v.y);
            q_shared[h][r][2*c4 + 1] = make_float2(v.z, v.w);
        }

        __syncthreads();

        // phase 4 - float products of the widened keys for the heads of this pass

#pragma unroll 8
        for (int c = 0; c < N_EMBD_H2; ++c) {
            float2 k_val[KEYS_PER_THREAD];
#pragma unroll
            for (int j = 0; j < KEYS_PER_THREAD; ++j) {
                k_val[j] = __half22float2(k_shared[kl + j*KEY_LANES][c]);
            }
#pragma unroll
            for (int h = 0; h < HEADS_PER_PASS; ++h) {
                const float2 q_val = q_shared[h][tl][c];
#pragma unroll
                for (int j = 0; j < KEYS_PER_THREAD; ++j) {
                    qk[h0 + h][j] = fmaf(k_val[j].x, q_val.x, qk[h0 + h][j]);
                    qk[h0 + h][j] = fmaf(k_val[j].y, q_val.y, qk[h0 + h][j]);
                }
            }
        }
    }

    // phase 5 - ReLU, weight, add the mask and write, consecutive threads write consecutive keys

    float score[KEYS_PER_THREAD] = { 0.0f };

#pragma unroll
    for (int h = 0; h < N_HEAD; ++h) {
#pragma unroll
        for (int j = 0; j < KEYS_PER_THREAD; ++j) {
            score[j] += fmaxf(qk[h][j], 0.0f) * w_shared[h][tl];
        }
    }

    const int i_batch = start_batch + tl;
    if (i_batch >= n_batch) {
        return;
    }

    const half * m_base = (const half *) ((const char *) M + i_batch*nbm1 + (i_stream%nem3)*nbm3);
    float * dst_base = (float *) ((char *) dst + i_batch*nb1 + i_stream*nb3);

#pragma unroll
    for (int j = 0; j < KEYS_PER_THREAD; ++j) {
        const int i_kv = start_kv + kl + j*KEY_LANES;
        if (i_kv < n_kv) {
            dst_base[i_kv] = score[j] + __half2float(m_base[i_kv]);
        }
    }
}

#define LIGHTNING_INDEXER_CASE(lightning_indexer_kernel, n_embd, n_head, K, type_K)         \
    if (K->type == (type_K)) {                                                              \
        lightning_indexer_kernel<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, n_embd, n_head, type_K> \
            <<<grid, block, 0, ctx.stream()>>>(                                             \
            q_d, k_d, w_d, m_d, dst_d,                                                      \
            n_stream, n_batch, n_kv,                                                        \
            nb1, nb2, nb3,                                                                  \
            nbq1, nbq2, nbq3,                                                               \
            nbk1, nbk2, nbk3,                                                               \
            nbw1, nbw2, nbw3,                                                               \
            nbm1, nbm2, nbm3,                                                               \
            nem3                                                                            \
        );                                                                                  \
    } else

// [T15] 同 LIGHTNING_INDEXER_CASE，但成功发射后直接 return。理由：宏链以 `} else` 结尾，
// 收尾语句就是它的 else 分支 —— 写 `break;` 会 switch 贯穿到下一个 case（每次调用多发射几个核，见 E-114），
// 写 `GGML_ABORT` 会在类型匹配后无条件 abort。只有 return 是对的（本函数在派发之后无其它语句）。
#define LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel, n_embd, n_head, K, type_K)    \
    if (K->type == (type_K)) {                                                              \
        if (lidx_launch_trace()) {                                                          \
            fprintf(stderr, "[lidx-launch] kvw=%d kvb=%d grid=%d n_batch=%d n_kv=%d\n",      \
                    kvw, (int) K_VECS_PER_BLOCK, (int) grid.x, (int) n_batch, (int) n_kv);   \
        }                                                                                   \
        lightning_indexer_kernel<WARPS_PER_BLOCK, K_VECS_PER_BLOCK, n_embd, n_head, type_K> \
            <<<grid, block, 0, ctx.stream()>>>(                                             \
            q_d, k_d, w_d, m_d, dst_d,                                                      \
            n_stream, n_batch, n_kv,                                                        \
            nb1, nb2, nb3,                                                                  \
            nbq1, nbq2, nbq3,                                                               \
            nbk1, nbk2, nbk3,                                                               \
            nbw1, nbw2, nbw3,                                                               \
            nbm1, nbm2, nbm3,                                                               \
            nem3                                                                            \
        );                                                                                  \
        return;                                                                             \
    } else


// [T15] decode 形状的核每 warp 处理几个池键。8（上游形态）会让 kernels 溢出到 scratch
// （VGPR 64 / Scratch_Size 144 B/线程），1 则完全不留 spill —— 实测 per-call 快 2.2×（PP8）、
// 6.2×（TP2×PP4，把那里的 3× 税清零），且贪心 md5 逐字节一致。宽容解析：未设/空/0/off/no/false ⇒ 用新默认 1，
// 只有显式 8 才回退到上游形态。

// [T15] 发射见证：设 GFX906_LIDX_TRACE_LAUNCH=1 时每次发射打一行（用来断言"每次调用只发一个核"）。
static bool lidx_launch_trace() {
    static const bool on = [] {
        const char * e = getenv("GFX906_LIDX_TRACE_LAUNCH");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return on;
}

static int lidx_decode_keys_per_warp() {
    const char * e = getenv("GFX906_LIDX_KVW");
    if (e == nullptr || e[0] == '\0' || strcmp(e, "0") == 0 || strcmp(e, "off") == 0 ||
        strcmp(e, "no") == 0 || strcmp(e, "false") == 0) {
        return 1;
    }
    const int v = atoi(e);
    return (v == 1 || v == 2 || v == 4 || v == 8) ? v : 1;
}

void ggml_cuda_lightning_indexer(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(  q->type == GGML_TYPE_F32);
    GGML_ASSERT(  w->type == GGML_TYPE_F32);
    GGML_ASSERT(  m->type == GGML_TYPE_F16);

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    // input tensor rows must be contiguous
    GGML_ASSERT(nbq0 == ggml_type_size(q->type));
    GGML_ASSERT(nbk0 == ggml_type_size(k->type));
    GGML_ASSERT(nbw0 == ggml_type_size(w->type));
    GGML_ASSERT(nbm0 == ggml_type_size(m->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int n_embd   = q->ne[0];
    const int n_head   = q->ne[1];
    const int n_batch  = q->ne[2];
    const int n_stream = q->ne[3];
    const int n_kv     = k->ne[2];

    const float *   q_d = (const float *)   q->data;
    const char  *   k_d = (const char  *)   k->data;
    const float *   w_d = (const float *)   w->data;
    const half  *   m_d = (const half  *)   m->data;
    float       * dst_d = (      float *) dst->data;

    const int device = ggml_cuda_get_device();
    const int cc     = ggml_cuda_info().devices[device].cc;

    if (n_embd == 128 && n_head == 64) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 64, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 64, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else if (n_embd == 128 && n_head == 32) {
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && turing_mma_available(cc) && k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_BF16) {
            // use wmma kernel
            constexpr int K_VECS_PER_BLOCK = 32;
            constexpr int WARPS_PER_BLOCK = 8;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_wmma, 128, 32, k, GGML_TYPE_Q8_0)
            GGML_ABORT("fatal error");
        } else {
#else // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
        {
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            // use vector kernel
            constexpr int K_VECS_PER_WARP = 8;
            constexpr int WARPS_PER_BLOCK = 8;
            constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

            dim3 block(32, WARPS_PER_BLOCK);
            int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
            dim3 grid(num_kv_blocks, n_batch, n_stream);

            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q4_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q5_1)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_Q8_0)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_BF16)
            LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 32, k, GGML_TYPE_F32)
            GGML_ABORT("fatal error");
        }
    } else if (n_embd == 128 && n_head == 4 && n_batch >= LIGHTNING_INDEXER_TILE_TOKENS) {
        // too few heads for a wmma tile, the tile kernel shares the keys across the tokens
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = 64;

        dim3 block(32, WARPS_PER_BLOCK);
        int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
        int num_batch_blocks = (n_batch + LIGHTNING_INDEXER_TILE_TOKENS - 1) / LIGHTNING_INDEXER_TILE_TOKENS;
        dim3 grid(num_kv_blocks, num_batch_blocks, n_stream);

        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F32)
        GGML_ABORT("fatal error");
    } else if (n_embd == 128 && n_head == 4) {
        // a batch smaller than a token tile, use vector kernel.
        // [T15] GFX906_LIDX_KVW 选每 warp 的键数：1（默认，无 spill）或 8（上游形态）等。
        // 两条分支的键数不同 ⇒ K_VECS_PER_BLOCK 与 grid 都跟着变，所以各写一遍。
        const int kvw = lidx_decode_keys_per_warp();
        switch (kvw) {
            case 1: {
                constexpr int K_VECS_PER_WARP = 1;
                constexpr int WARPS_PER_BLOCK = 8;
                constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

                dim3 block(32, WARPS_PER_BLOCK);
                int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
                dim3 grid(num_kv_blocks, n_batch, n_stream);

                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F16)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_0)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_1)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_0)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_1)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q8_0)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_BF16)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F32)
                GGML_ABORT("fatal error");
            }
            case 2: {
                constexpr int K_VECS_PER_WARP = 2;
                constexpr int WARPS_PER_BLOCK = 8;
                constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

                dim3 block(32, WARPS_PER_BLOCK);
                int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
                dim3 grid(num_kv_blocks, n_batch, n_stream);

                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F16)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_0)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_1)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_0)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_1)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q8_0)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_BF16)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F32)
                GGML_ABORT("fatal error");
            }
            case 4: {
                constexpr int K_VECS_PER_WARP = 4;
                constexpr int WARPS_PER_BLOCK = 8;
                constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

                dim3 block(32, WARPS_PER_BLOCK);
                int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
                dim3 grid(num_kv_blocks, n_batch, n_stream);

                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F16)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_0)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_1)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_0)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_1)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q8_0)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_BF16)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F32)
                GGML_ABORT("fatal error");
            }
            default: {
                constexpr int K_VECS_PER_WARP = 8;
                constexpr int WARPS_PER_BLOCK = 8;
                constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

                dim3 block(32, WARPS_PER_BLOCK);
                int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
                dim3 grid(num_kv_blocks, n_batch, n_stream);

                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F16)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_0)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_1)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_0)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_1)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q8_0)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_BF16)
                LIGHTNING_INDEXER_CASE_RET(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F32)
                GGML_ABORT("fatal error");
            }
        }
    } else {
        GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_lightning_indexer_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2]; // weights
    const ggml_tensor * m = dst->src[3]; // mask

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)
    GGML_TENSOR_LOCALS(int64_t, ne, dst, ne)
    GGML_TENSOR_LOCALS(size_t,  nb, dst, nb)

    if (neq0 != 128) {
        return false;
    }

    if (neq1 != 64 && neq1 != 32 && neq1 != 4) {
        return false;
    }

    // alignment checks
    for (const ggml_tensor * t : {q, k}) {
        if (ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }

    switch(k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}

// ---------------------------------------------------------------------------
// QSA lightning-indexer score + top-k fusion (GGML_OP_LIDX_TOPK_CHUNK / _MERGE)
//
// The score of every pool cell is computed with the very same tile/vec kernels as
// ggml_lightning_indexer, but restricted to a chunk of the pool: each chunk is
// scored into the op's scratch region and immediately reduced to a per-chunk
// top-N_TOP table of (score, pool index) pairs. A second op merges the chunk
// tables. This avoids materializing the full [n_pool, n_tokens] f32 score matrix
// (multi-GiB at long context) in the compute buffer.
//
// Selection semantics: scores identical to ggml_lightning_indexer; the top-N_TOP
// set is deterministic (score descending, dead padding last, then lower pool
// index first) - exact score ties may be ordered differently than ggml_top_k.
// ---------------------------------------------------------------------------

static __device__ __forceinline__ bool ldxtk_pair_before(const float2 & a, const float2 & b) {
    if (a.x != b.x) {
        return a.x > b.x;
    }
    const bool da = a.y < 0.0f;
    const bool db = b.y < 0.0f;
    if (da != db) {
        return !da;
    }
    return a.y < b.y;
}

// insertion sort of N register pairs in ldxtk order (best first)
template <int N>
static __device__ void ldxtk_sort_regs(float2 * a) {
#pragma unroll
    for (int i = 1; i < N; ++i) {
        float2 v = a[i];
        int j = i;
        while (j > 0 && ldxtk_pair_before(v, a[j - 1])) {
            a[j] = a[j - 1];
            --j;
        }
        a[j] = v;
    }
}

// float -> orderable uint32 (larger float maps to larger uint; -inf -> smallest)
static __device__ __forceinline__ uint32_t ldxtk_ord(const float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// descending value-only bitonic sort of N shared floats (256 threads)
template <int N>
static __device__ void ldxtk_bitonic_sh_f(float * s, const int tl) {
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tl; i < N; i += 256) {
                const int p = i ^ j;
                if (p > i) {
                    float & a = s[i];
                    float & b = s[p];
                    const bool up = ((i & k) == 0);
                    if (up ? (b > a) : (a > b)) {
                        const float t = a; a = b; b = t;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// bitonic sort of N shared pairs in ldxtk order (best first), 256 threads
// (full bitonic network: every (i, i^j) pair is exchanged by its lower index;
// the loop must cover all N positions, not just N/2)
template <int N>
static __device__ void ldxtk_bitonic_sh(float2 * s, const int tl) {
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tl; i < N; i += 256) {
                const int p = i ^ j;
                if (p > i) {
                    float2 & a = s[i];
                    float2 & b = s[p];
                    const bool up = ((i & k) == 0);
                    if (up ? ldxtk_pair_before(b, a) : ldxtk_pair_before(a, b)) {
                        float2 t = a; a = b; b = t;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// fill the scratch tail of a chunk (positions [n_valid, chunk)) with -inf so the
// scratch region is fully defined (the CPU reference writes -inf there too)
static __global__ void ldxtk_scratch_fill_kernel(float * scratch, int64_t n_valid, int64_t chunk, int64_t n_tokens) {
    const int64_t t = blockIdx.x;
    for (int64_t i = n_valid + threadIdx.x; i < chunk; i += 256) {
        scratch[t * chunk + i] = -INFINITY;
    }
}

// One block (256 threads) per (token, sub-chunk). Each thread holds 32
// candidates (real or dead padding); the 32 are sorted once in registers and
// the full top-N_TOP of the block's <= 8192 candidates is selected with a
// shared-memory bitonic sort (64 KiB).
//
// MODE 0 (SCORE):  candidates are chunk scores in a scratch row (src_s),
//                  idx = kv0 + pos; the N_TOP pairs are written contiguously.
// MODE 1 (TABLE):  candidates are (score, idx) pairs of a table stack
//                  [n_tables, n_tokens, N_TOP] (src_p, idx already global);
//                  the N_TOP pairs are written contiguously.
// MODE 2 (SPLIT):  like TABLE, but the pairs are split into two strided planes
//                  (out = score plane, out1 = index plane, stride n_tokens).
template <int N_TOP, int MODE>
static __global__ void __launch_bounds__(256, 1) lidxtk_reduce_kernel(
        const float * src_s,    // MODE 0: scratch base, token row stride src_stride
        const float * src_p,    // MODE 1/2: table base (token row stride N_TOP)
        int64_t n_cand,         // <= 8192
        int64_t n_tokens,
        float * out,            // MODE 0/1: contiguous pairs; MODE 2: score plane (float)
        float * out1,           // MODE 2: index plane (float)
        int64_t kv0,            // MODE 0: global index offset
        int64_t src_stride,     // MODE 0: floats per token row
        int64_t tab_stride) {   // MODE 1: float2 per table (n_tokens * N_TOP)

    constexpr int N_PER_THREAD = 32;
    constexpr int N_CAND_MAX   = 256 * N_PER_THREAD;  // 8192

    const int tl = threadIdx.x;
    const int t  = blockIdx.x;

    const float2 DEAD = make_float2(-INFINITY, -1.0f);

    const float2 * src_p_t = (const float2 *) src_p + (MODE == 0 ? 0 : (size_t) t * N_TOP);
    const float * src_s_t = src_s + t * src_stride;

    float2 keep[N_PER_THREAD];
#pragma unroll
    for (int i = 0; i < N_PER_THREAD; ++i) {
        const int p = tl + i * 256;
        if (p < n_cand) {
            if (MODE == 0) {
                keep[i] = make_float2(src_s_t[p], (float) (kv0 + p));
            } else if (MODE == 2) {
                keep[i] = src_p_t[p];
            } else {
                // MODE 1: table stack [n_tables, n_tokens, N_TOP] - table-major
                keep[i] = src_p_t[(p / N_TOP) * tab_stride + (p % N_TOP)];
            }
        } else {
            keep[i] = DEAD;
        }
    }

    ldxtk_sort_regs<N_PER_THREAD>(keep);

    __shared__ float2 sh[N_CAND_MAX];  // 64 KiB

#pragma unroll
    for (int i = 0; i < N_PER_THREAD; ++i) {
        sh[tl * N_PER_THREAD + i] = keep[i];
    }
    __syncthreads();

    ldxtk_bitonic_sh<N_CAND_MAX>(sh, tl);

#pragma unroll
    for (int i = tl; i < N_TOP; i += 256) {
        const float2 v = sh[i];
        if (MODE == 2) {
            // the merge output is [2, n_top, n_tokens]: the slot index is the fastest axis,
            // so that a [n_top, n_tokens] view of the plane reads the slots of one token
            out[t * N_TOP + i]  = v.x;
            out1[t * N_TOP + i] = v.y;
        } else {
            ((float2 *) out)[(size_t) t * N_TOP + i] = v;
        }
    }
}

// Radix top-N_TOP for large candidate sets (n_cand > 2048). The old kernel's full
// bitonic sort of 8192 pairs (91 barrier stages, 64 KiB, ~50 us per block on MI50)
// was the bottleneck of the fused path; this narrows the candidates with shared
// radix rounds (10+8+8+6 bits) instead, then finalizes with a small bitonic:
//
//   pass 1:  load the n_cand values into shared
//   rounds:  histogram the top bits, find the cutoff bin, gather survivors in place
//   final:   sort the <= 1024 survivors, threshold = N_TOP-th value, then re-scan
//            the global input keeping {v > thr} plus the first (N_TOP - A) ties in
//            scan order (which also matches the reference's lower-index-first ties)
//
// MODE 0 (SCORE): values from the scratch row, idx = kv0 + pos.
// MODE 1 (TABLE): (score, idx) pairs of a table stack [n_tables, n_tokens, N_TOP],
//                 idx already global, strided by tab_stride float2 per table.
// MODE 2 (SPLIT): like TABLE, but the output is split into two strided planes.
template <int N_TOP, int MODE>
static __global__ void __launch_bounds__(256, 1) lidxtk_reduce_radix_kernel(
        const float * src_s,    // MODE 0: scratch base, token row stride src_stride
        const float * src_p,    // MODE 1/2: table base (token row stride N_TOP)
        int64_t n_cand,         // <= 8192
        int64_t n_tokens,
        float * out,            // MODE 0/1: contiguous pairs; MODE 2: score plane (float)
        float * out1,           // MODE 2: index plane (float)
        int64_t kv0,            // MODE 0: global index offset
        int64_t src_stride,     // MODE 0: floats per token row
        int64_t tab_stride) {   // MODE 1: float2 per table (n_tokens * N_TOP)

    constexpr int N_CAND_MAX = 8192;

    const int tl = threadIdx.x;
    const int t  = blockIdx.x;

    const float2 DEAD = make_float2(-INFINITY, -1.0f);

    const float2 * src_p_t = (const float2 *) src_p + (MODE == 0 ? 0 : (size_t) t * N_TOP);
    const float * src_s_t  = src_s + t * src_stride;

    __shared__ float    sv[N_CAND_MAX];   // 32 KiB
    __shared__ uint32_t hist[1032];       // 1024 bins + 4 front + 4 safe wavefront totals
    __shared__ float2   keep[1024];       // 8 KiB
    __shared__ uint8_t  tie_mark[N_CAND_MAX];  // 8 KiB
    __shared__ float    warp_min[4];      // 256 threads = 4 wavefronts (64 lanes)
    __shared__ int      s_m, s_cut, s_above, s_keep, s_ties;
    __shared__ float    s_min;

    auto val_at = [&](int i) -> float {
        if (MODE == 0) {
            return src_s_t[i];
        } else if (MODE == 2) {
            return src_p_t[i].x;
        } else {
            return src_p_t[(i / N_TOP) * tab_stride + (i % N_TOP)].x;
        }
    };
    auto idx_at = [&](int i) -> float {
        if (MODE == 0) {
            return (float) (kv0 + i);
        } else if (MODE == 2) {
            return src_p_t[i].y;
        } else {
            return src_p_t[(i / N_TOP) * tab_stride + (i % N_TOP)].y;
        }
    };

    if (n_cand == 0) {
        // all-dead table
#pragma unroll
        for (int i = tl; i < N_TOP; i += 256) {
            if (MODE == 2) {
                out[i * n_tokens + t]  = -INFINITY;
                out1[i * n_tokens + t] = -1.0f;
            } else {
                ((float2 *) out)[(size_t) t * N_TOP + i] = DEAD;
            }
        }
        return;
    }

    // pass 1: load values
    for (int i = tl; i < n_cand; i += 256) {
        sv[i] = val_at(i);
    }
    __syncthreads();

    // Radix rounds: 10 + 8 + 8 + 6 bits.
    // State: sv[0..f) = frontier (still to be refined), sv[f..m) = safe (definitely
    // in the top N_TOP). Each round re-buckets the FRONTIER only, by the next bits
    // (valid because all frontier values share the bits above): bins strictly above
    // the cutoff join the safe region, exactly the cutoff bin becomes the new
    // frontier, the rest is dropped. (Bucketsing the whole survivor set instead
    // would order values of different upper bins wrongly across the next bits.)
    // Race-free gather: the frontier is loaded into registers first, and only then
    // is shared memory written.
    int m = n_cand;
    int f = n_cand;
    constexpr int shifts[4]    = {22, 14, 6, 0};
    constexpr int nbins_arr[4] = {1024, 256, 256, 64};
    float r[32];
    float rm[32];  // staging for the safe-region move
    const int wf = tl >> 6;
    const int lane = tl & 63;
    #pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        const int safe_n = m - f;
        const int need   = N_TOP - safe_n;  // values still to be taken from the frontier
        if (need <= 0) break;               // the safe region alone fills the top
        if (f <= need) break;               // the whole frontier is selected
        const int nbins = nbins_arr[rr];
        const int shift = shifts[rr];
        const int n_local = std::min(32, f - tl * 32);  // may be <= 0
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (i < n_local) r[i] = sv[tl * 32 + i];
        }
        for (int b = tl; b < nbins; b += 256) hist[b] = 0;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (i < n_local) {
                atomicAdd(&hist[(ldxtk_ord(r[i]) >> shift) & (nbins - 1)], 1u);
            }
        }
        __syncthreads();
        // cutoff: T = f - need frontier values are not selected; cut = the first bin
        // (ascending) with cumulative C[bin] > T. One full wavefront (64 lanes on
        // gfx906) handles the nbins/64 slices; the lanes with x <= T are a monotone
        // prefix, so its length is the popcount of the ballot mask.
        if (tl < 64) {
            const int slice = nbins / 64;
            uint32_t c = 0;
            for (int b = tl * slice; b < (tl + 1) * slice; ++b) c += hist[b];
            uint32_t x = c;  // inclusive prefix of slice sums
#pragma unroll
            for (int d = 1; d < 64; d <<= 1) {
                const uint32_t y = __shfl_up_sync(0xffffffffffffffffull, x, d, 64);
                if (tl >= d) x += y;
            }
            const uint32_t T = (uint32_t) (f - need);
            const uint64_t mask = __ballot_sync(0xffffffffffffffffull, x <= T);
            const int cross = (mask == 0) ? 0 : (int) __popcll(mask);
            if (tl == cross) {
                uint32_t cc = (cross == 0) ? 0 : (x - c);  // prefix before this slice
                int b = cross * slice;
                for (int i = 0; i < slice; ++i) {
                    cc += hist[cross * slice + i];
                    if (cc > T) {
                        b = cross * slice + i;
                        break;
                    }
                }
                s_cut = b;
            }
        }
        __syncthreads();
        const int cut = s_cut;
        // per-thread counts: new frontier (bin == cut) and new safe (bin > cut)
        int n_front = 0, n_safe = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (i < n_local) {
                const uint32_t b = (ldxtk_ord(r[i]) >> shift) & (nbins - 1);
                if (b == (uint32_t) cut) ++n_front;
                else if (b > (uint32_t) cut) ++n_safe;
            }
        }
        // exclusive block prefixes of n_front and n_safe (wavefront prefix +
        // cross-wavefront totals). Both chains are shuffle-only; the totals are
        // published to disjoint shared slots after a single barrier (reusing one
        // slot range for both would race: the second write can land before other
        // wavefronts finished reading the first).
        int xf = n_front;
#pragma unroll
        for (int d = 1; d < 64; d <<= 1) {
            const int y = __shfl_up_sync(0xffffffffffffffffull, xf, d, 64);
            if (lane >= d) xf += y;
        }
        int xs = n_safe;
#pragma unroll
        for (int d = 1; d < 64; d <<= 1) {
            const int y = __shfl_up_sync(0xffffffffffffffffull, xs, d, 64);
            if (lane >= d) xs += y;
        }
        if (lane == 63) {
            hist[1024 + wf] = (uint32_t) xf;
            hist[1028 + wf] = (uint32_t) xs;
        }
        __syncthreads();
        int base_front = xf - n_front;
        if (wf > 0) base_front += (int) hist[1024 + wf - 1];
        if (wf > 1) base_front += (int) hist[1024 + wf - 2];
        if (wf > 2) base_front += (int) hist[1024 + wf - 3];
        const int f2 = (int) (hist[1024] + hist[1025] + hist[1026] + hist[1027]);
        int base_safe = xs - n_safe;
        if (wf > 0) base_safe += (int) hist[1028 + wf - 1];
        if (wf > 1) base_safe += (int) hist[1028 + wf - 2];
        if (wf > 2) base_safe += (int) hist[1028 + wf - 3];
        const int s2 = (int) (hist[1028] + hist[1029] + hist[1030] + hist[1031]);
        // move the safe region [f, m) to [f2, f2 + safe_n). The copy overlaps
        // itself (f2 <= f) and a strided multi-thread copy has no cross-thread
        // ordering, so stage the slice in registers first.
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int j = tl + i * 256;
            if (j < safe_n) rm[i] = sv[f + j];
        }
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int j = tl + i * 256;
            if (j < safe_n) sv[f2 + j] = rm[i];
        }
        __syncthreads();
        // gather: new frontier -> [0, f2), safe_new -> [f2 + safe_n, ...)
        int cf = 0, cs = 0;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            if (i < n_local) {
                const uint32_t b = (ldxtk_ord(r[i]) >> shift) & (nbins - 1);
                if (b == (uint32_t) cut) {
                    sv[base_front + cf] = r[i];
                    ++cf;
                } else if (b > (uint32_t) cut) {
                    sv[f2 + safe_n + base_safe + cs] = r[i];
                    ++cs;
                }
            }
        }
        __syncthreads();
        f = f2;
        m = f2 + safe_n + s2;
    }
    __syncthreads();

    // threshold for the keep pass: the minimum of the selected values. The
    // selection is the safe region plus (the whole frontier, or `need` of its
    // bit-identical members after 4 rounds); in either case the minimum of the
    // frontier (or of the whole safe region when it alone fills the top) is it.
    float thr;
    {
        const int safe_n = m - f;
        const int lo = (m < N_TOP || N_TOP - safe_n > 0) ? 0 : f;
        const int hi = (m < N_TOP || N_TOP - safe_n > 0) ? f : m;
        float mn = INFINITY;
        for (int i = tl; i < hi - lo; i += 256) mn = mn < sv[lo + i] ? mn : sv[lo + i];
        for (int d = 32; d > 0; d >>= 1) {
            const float y = __shfl_xor_sync(0xffffffffffffffffull, mn, d, 64);
            mn = mn < y ? mn : y;
        }
        if ((tl & 63) == 0) {
            warp_min[tl >> 6] = mn;
        }
        __syncthreads();
        if (tl == 0) {
            float x = INFINITY;
#pragma unroll
            for (int w = 0; w < 4; ++w) x = x < warp_min[w] ? x : warp_min[w];
            s_min = x;
        }
        __syncthreads();
        thr = s_min;
    }
    __syncthreads();

    // keep pass: collect the {v > thr} pairs plus the lowest-index `need` of the
    // {v == thr} ties (matching the CPU reference's index-ordered tie-breaking).
    s_above = 0;
    s_ties = 0;
    __syncthreads();
    for (int i = tl; i < n_cand; i += 256) {
        const float v = val_at(i);
        tie_mark[i] = 0;
        if (v > thr) {
            atomicAdd(&s_above, 1);
        } else if (v == thr) {
            tie_mark[i] = 1;
            atomicAdd(&s_ties, 1);
        }
    }
    __syncthreads();
    const int above = s_above;
    int need = N_TOP - above;
    if (need < 0) need = 0;

    s_keep = 0;
    __syncthreads();
    for (int i = tl; i < n_cand; i += 256) {
        const float v = val_at(i);
        if (v > thr) {
            const int j = atomicAdd(&s_keep, 1);
            if (j < 1024) keep[j] = make_float2(v, idx_at(i));
        }
    }
    __syncthreads();
    // the ties: keep the lowest-index `need`. Each thread owns a contiguous 32-slot
    // chunk (256 x 32 = 8192 >= n_cand) so the block prefix over the per-chunk tie
    // counts is a true index-order prefix.
    if (need > 0) {
        const int i0 = tl * 32;
        uint32_t c = 0;
        #pragma unroll
        for (int c2 = 0; c2 < 32; ++c2) {
            const int i = i0 + c2;
            c += (i < n_cand && tie_mark[i]) ? 1u : 0u;
        }
        int x = (int) c;
#pragma unroll
        for (int d = 1; d < 64; d <<= 1) {
            const int y = __shfl_up_sync(0xffffffffffffffffull, x, d, 64);
            if (lane >= d) x += y;
        }
        if (lane == 63) hist[1024 + wf] = (uint32_t) x;
        __syncthreads();
        int base = x - (int) c;
        if (wf > 0) base += (int) hist[1024 + wf - 1];
        if (wf > 1) base += (int) hist[1024 + wf - 2];
        if (wf > 2) base += (int) hist[1024 + wf - 3];
        int lr = 0;
        #pragma unroll
        for (int c2 = 0; c2 < 32; ++c2) {
            const int i = i0 + c2;
            if (i < n_cand && tie_mark[i]) {
                if (base + lr < need && above + base + lr < 1024) keep[above + base + lr] = make_float2(thr, idx_at(i));
                ++lr;
            }
        }
    }
    __syncthreads();

    // pad and sort the kept pairs, then write the top N_TOP
    const int n_keep = std::min(above + std::min(need, (int) s_ties), 1024);
    for (int i = n_keep + tl; i < 1024; i += 256) keep[i] = DEAD;
    __syncthreads();
    ldxtk_bitonic_sh<1024>(keep, tl);
    __syncthreads();
#pragma unroll
    for (int i = tl; i < N_TOP; i += 256) {
        const float2 v = keep[i];
        if (MODE == 2) {
            // the merge output is [2, n_top, n_tokens]: the slot index is the fastest axis,
            // so that a [n_top, n_tokens] view of the plane reads the slots of one token
            out[t * N_TOP + i]  = v.x;
            out1[t * N_TOP + i] = v.y;
        } else {
            ((float2 *) out)[(size_t) t * N_TOP + i] = v;
        }
    }
}

#define LDXTK_LAUNCH_REDUCER(MODE)                                                                              \
    static void ldxtk_launch_reducer_##MODE(                                                                    \
            int n_top, const float * src_s, const float * src_p, int64_t n_cand,                                \
            int64_t n_tokens, float * out, float * out1, int64_t kv0, int64_t src_stride, int64_t tab_stride,  \
            cudaStream_t stream) {                                                                              \
        if (n_top == 256) {                                                                                     \
            if (n_cand > 2048) {                                                                               \
                lidxtk_reduce_radix_kernel<256, MODE>  <<<n_tokens, 256, 0, stream>>>(src_s, src_p, n_cand, n_tokens, out, out1, kv0, src_stride, tab_stride); \
            } else {                                                                                           \
                lidxtk_reduce_kernel<256, MODE>  <<<n_tokens, 256, 0, stream>>>(src_s, src_p, n_cand, n_tokens, out, out1, kv0, src_stride, tab_stride); \
            }                                                                                                  \
            return;                                                                                             \
        }                                                                                                       \
        if (n_top == 512) {                                                                                     \
            if (n_cand > 2048) {                                                                               \
                lidxtk_reduce_radix_kernel<512, MODE>  <<<n_tokens, 256, 0, stream>>>(src_s, src_p, n_cand, n_tokens, out, out1, kv0, src_stride, tab_stride); \
            } else {                                                                                           \
                lidxtk_reduce_kernel<512, MODE>  <<<n_tokens, 256, 0, stream>>>(src_s, src_p, n_cand, n_tokens, out, out1, kv0, src_stride, tab_stride); \
            }                                                                                                  \
            return;                                                                                             \
        }                                                                                                       \
        if (n_top == 1024) {                                                                                    \
            if (n_cand > 2048) {                                                                               \
                lidxtk_reduce_radix_kernel<1024, MODE> <<<n_tokens, 256, 0, stream>>>(src_s, src_p, n_cand, n_tokens, out, out1, kv0, src_stride, tab_stride); \
            } else {                                                                                           \
                lidxtk_reduce_kernel<1024, MODE> <<<n_tokens, 256, 0, stream>>>(src_s, src_p, n_cand, n_tokens, out, out1, kv0, src_stride, tab_stride); \
            }                                                                                                  \
            return;                                                                                             \
        }                                                                                                       \
        GGML_ABORT("ldxtk: unsupported n_top");                                                                 \
    }

LDXTK_LAUNCH_REDUCER(0)
LDXTK_LAUNCH_REDUCER(1)
LDXTK_LAUNCH_REDUCER(2)

// score one pool chunk with the plain tile/vec kernel into the scratch row
// layout [n_tokens, chunk] (dst row stride nb1 = chunk, nb3 = 0)
static void ggml_cuda_ldxtk_score_chunk(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * dst_blob,
        int64_t c) {

    const ggml_tensor * q = dst_blob->src[0];
    const ggml_tensor * k = dst_blob->src[1];
    const ggml_tensor * w = dst_blob->src[2]; // weights
    const ggml_tensor * m = dst_blob->src[3]; // mask

    const int32_t chunk = dst_blob->op_params[1];

    GGML_TENSOR_LOCALS(int64_t, neq,  q, ne)
    GGML_TENSOR_LOCALS(size_t,  nbq,  q, nb)
    GGML_TENSOR_LOCALS(int64_t, nek,  k, ne)
    GGML_TENSOR_LOCALS(size_t,  nbk,  k, nb)
    GGML_TENSOR_LOCALS(int64_t, new,  w, ne)
    GGML_TENSOR_LOCALS(size_t,  nbw,  w, nb)
    GGML_TENSOR_LOCALS(int64_t, nem,  m, ne)
    GGML_TENSOR_LOCALS(size_t,  nbm,  m, nb)

    const int n_stream = 1;
    const int n_batch  = q->ne[2];
    const int n_kv     = std::min<int64_t>(chunk, k->ne[2] - c * chunk);
    GGML_ASSERT(n_kv > 0);

    const float * q_d   = (const float *) q->data;
    const char  * k_d   = (const char *)  k->data + c * chunk * nbk2;
    const float * w_d   = (const float *) w->data;
    const half  * m_d   = (const half *)  m->data + c * chunk;
    float       * dst_d = (float *) ((char *) dst_blob->data);

    // dst scratch rows are [n_tokens, chunk]; the kernel strides rows by nb1
    const size_t nb1 = (size_t) chunk * sizeof(float);
    const size_t nb2 = 0;
    const size_t nb3 = 0;

    const int n_embd = q->ne[0];
    const int n_head = q->ne[1];
    GGML_ASSERT(n_embd == 128 && n_head == 4);

    if (n_batch >= LIGHTNING_INDEXER_TILE_TOKENS) {
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = 64;

        dim3 block(32, WARPS_PER_BLOCK);
        int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
        int num_batch_blocks = (n_batch + LIGHTNING_INDEXER_TILE_TOKENS - 1) / LIGHTNING_INDEXER_TILE_TOKENS;
        dim3 grid(num_kv_blocks, num_batch_blocks, n_stream);

        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_tile, 128, 4, k, GGML_TYPE_F32)
        GGML_ABORT("fatal error");
    } else {
        constexpr int K_VECS_PER_WARP = 8;
        constexpr int WARPS_PER_BLOCK = 8;
        constexpr int K_VECS_PER_BLOCK = K_VECS_PER_WARP * WARPS_PER_BLOCK;

        dim3 block(32, WARPS_PER_BLOCK);
        int num_kv_blocks = (n_kv + (K_VECS_PER_BLOCK) - 1) / (K_VECS_PER_BLOCK);
        dim3 grid(num_kv_blocks, n_batch, n_stream);

        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q4_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q5_1)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_Q8_0)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_BF16)
        LIGHTNING_INDEXER_CASE(lightning_indexer_kernel_vec, 128, 4, k, GGML_TYPE_F32)
        GGML_ABORT("fatal error");
    }
}

// the reduce kernel handles at most 256*32 = 8192 candidates per block; longer
// chunks are split into sub-chunks, each producing its own partial table
constexpr int64_t LDXTK_SUBCHUNK = 8192;

void ggml_cuda_ldxtk_chunk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];

    const int32_t n_top  = dst->op_params[0];
    const int32_t chunk  = dst->op_params[1];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(n_top == 256 || n_top == 512 || n_top == 1024);
    GGML_ASSERT(chunk <= 2 * LDXTK_SUBCHUNK);

    const int64_t n_batch  = q->ne[2];
    const int64_t n_kv     = k->ne[2];
    const int64_t n_chunks = (n_kv + chunk - 1) / chunk;
    const int64_t n_half   = (chunk + LDXTK_SUBCHUNK - 1) / LDXTK_SUBCHUNK;

    float * scratch = (float *) dst->data;                // [n_tokens, chunk]
    float * partial = scratch + n_batch * chunk;          // [n_chunks*n_half, n_tokens, n_top, 2]

    cudaStream_t stream = ctx.stream();

    for (int64_t c = 0; c < n_chunks; ++c) {
        const int64_t kvn = std::min<int64_t>(chunk, n_kv - c * chunk);

        ggml_cuda_ldxtk_score_chunk(ctx, dst, c);
        CUDA_CHECK(cudaGetLastError());

        if (kvn < chunk) {
            ldxtk_scratch_fill_kernel<<<n_batch, 256, 0, stream>>>(scratch, kvn, chunk, n_batch);
            CUDA_CHECK(cudaGetLastError());
        }

        for (int64_t h = 0; h < n_half; ++h) {
            const int64_t off  = h * LDXTK_SUBCHUNK;
            const int64_t kvn_h = std::min<int64_t>(LDXTK_SUBCHUNK, kvn - off);
            float * out_c = partial + (c * n_half + h) * n_batch * n_top * 2;
            // kvn_h == 0 launches an all-dead table so every slot is defined
            ldxtk_launch_reducer_0(
                n_top, scratch + off, nullptr, std::max<int64_t>(0, kvn_h), n_batch, out_c, nullptr, c * chunk + off, chunk, 0, stream);
            CUDA_CHECK(cudaGetLastError());
        }
    }
}

void ggml_cuda_ldxtk_merge(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * partial = dst->src[0];
    const ggml_tensor * q       = partial->src[0];
    const ggml_tensor * k       = partial->src[1];

    const int32_t n_top  = partial->op_params[0];
    const int32_t chunk  = partial->op_params[1];

    GGML_ASSERT(n_top == 256 || n_top == 512 || n_top == 1024);

    const int64_t n_batch  = q->ne[2];
    const int64_t n_kv     = k->ne[2];
    const int64_t n_chunks = (n_kv + chunk - 1) / chunk;

    float * partial_f = (float *) partial->data;
    const size_t n_scratch = (size_t) n_batch * chunk;
    float2 * tables = (float2 *) (partial_f + n_scratch); // [n_chunks*n_half, n_tokens, n_top, 2]
    float2 * temp   = (float2 *) partial_f;               // one table in the scratch region

    const size_t table_floats = (size_t) n_batch * n_top * 2;
    GGML_ASSERT(n_scratch >= table_floats && "ldxtk: scratch too small for a merge table");

    const int64_t n_half   = (chunk + LDXTK_SUBCHUNK - 1) / LDXTK_SUBCHUNK;
    int64_t n_tab          = n_chunks * n_half;
    const int64_t n_group  = LDXTK_SUBCHUNK / n_top;  // tables per launch (<= 8192 candidates)

    cudaStream_t stream = ctx.stream();

    // one launch merges at most n_group tables (<= 8192 candidates per block);
    // reduce more in rounds, reusing the scratch region for the merged table.
    // table n_out never aliases an input of the current round (n_out < c0).
    while (n_tab > 1) {
        int64_t n_out = 0;
        for (int64_t c0 = 0; c0 < n_tab; c0 += n_group) {
            const int64_t n_in = std::min<int64_t>(n_group, n_tab - c0);
            ldxtk_launch_reducer_1(
                n_top, nullptr, (float *) (tables + c0 * n_batch * n_top), n_in * n_top,
                n_batch, (float *) temp, nullptr, 0, 0, n_batch * n_top, stream);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(
                (float *) (tables + n_out * n_batch * n_top), (float *) temp,
                table_floats * sizeof(float), cudaMemcpyDeviceToDevice, stream));
            ++n_out;
        }
        n_tab = n_out;
    }

    float * out0 = (float *) dst->data;
    float * out1 = (float *) dst->data + (size_t) n_top * n_batch;
    ldxtk_launch_reducer_2(n_top, nullptr, (float *) tables, n_top, n_batch, out0, out1, 0, 0, 0, stream);
    CUDA_CHECK(cudaGetLastError());
}

bool ggml_cuda_ldxtk_chunk_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);

    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * m = dst->src[3]; // mask

    const int32_t n_top = dst->op_params[0];

    if (n_top != 256 && n_top != 512 && n_top != 1024) {
        return false;
    }

    if (q->ne[0] != 128 || q->ne[1] != 4) {
        return false;
    }

    // single stream only
    if (q->ne[3] != 1 || k->ne[3] != 1 || dst->src[2]->ne[3] != 1 || m->ne[3] != 1) {
        return false;
    }

    // alignment checks
    for (const ggml_tensor * t : {q, k}) {
        if (ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                return false;
            }
        }
    }

    switch(k->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q4_0:
            return true;
        default:
            return false;
    }
}

bool ggml_cuda_ldxtk_merge_supported(int device, const ggml_tensor * dst) {
    const ggml_tensor * partial = dst->src[0];
    if (partial->op != GGML_OP_LIDX_TOPK_CHUNK) {
        return false;
    }
    return ggml_cuda_ldxtk_chunk_supported(device, partial);
}
