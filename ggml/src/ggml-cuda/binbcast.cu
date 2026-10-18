#include "binbcast.cuh"
#include "convert.cuh"
#include "unary.cuh"
#include <cstdint>
#include <utility>

template<typename T, size_t>
using type_for_index = T;

static __device__ __forceinline__ float op_repeat(const float a, const float b) {
    return b;
    GGML_UNUSED(a);
}

static __device__ __forceinline__ float op_add(const float a, const float b) {
    return a + b;
}

static __device__ __forceinline__ float op_sub(const float a, const float b) {
    return a - b;
}

static __device__ __forceinline__ float op_mul(const float a, const float b) {
    return a * b;
}

static __device__ __forceinline__ float op_div(const float a, const float b) {
    return a / b;
}

template <float (*bin_op)(const float, const float),
          typename src0_t,
          typename src1_t,
          typename dst_t,
          typename... src1_ptrs>
static __global__ void k_bin_bcast(const src0_t *         src0,
                                   const src1_t *         src1,
                                   dst_t *                dst,
                                   const uint32_t         ne0,
                                   const uint32_t         ne1,
                                   const uint32_t         ne2,
                                   const uint3            ne3,
                                   const uint3            ne10,
                                   const uint3            ne11,
                                   const uint3            ne12,
                                   const uint3            ne13,
                                 /*const uint32_t         s0,*/
                                   const uint32_t         s1,
                                   const uint32_t         s2,
                                   const uint32_t         s3,
                                   const uint32_t         s00,
                                   const uint32_t         s01,
                                   const uint32_t         s02,
                                   const uint32_t         s03,
                                   const uint32_t         s10,
                                   const uint32_t         s11,
                                   const uint32_t         s12,
                                   const uint32_t         s13,
                                   src1_ptrs... src1s) {
    ggml_cuda_pdl_lc();
    const uint32_t i0s = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t i1  = (blockDim.y * blockIdx.y + threadIdx.y);
    const uint32_t i2  = fastdiv((blockDim.z * blockIdx.z + threadIdx.z), ne3);
    const uint32_t i3  = (blockDim.z * blockIdx.z + threadIdx.z) - (i2 * ne3.z);

    if (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3.z) {
        return;
    }

    const uint32_t i11 = fastmodulo(i1, ne11);
    const uint32_t i12 = fastmodulo(i2, ne12);
    const uint32_t i13 = fastmodulo(i3, ne13);

    const size_t i_src0 = size_t( i3)*s03 + size_t( i2)*s02 + size_t( i1)*s01;
    const size_t i_src1 = size_t(i13)*s13 + size_t(i12)*s12 + size_t(i11)*s11;
    const size_t i_dst  = size_t( i3)*s3  + size_t( i2)*s2  + size_t( i1)*s1;

    const src0_t * src0_row = src0 ? (src0 + i_src0) : nullptr;
    dst_t * dst_row = dst + i_dst;

    const uint32_t s0 = blockDim.x * gridDim.x;

    ggml_cuda_pdl_sync();
    for (uint32_t i0 = i0s; i0 < ne0; i0 += s0) {
        const uint32_t i10 = fastmodulo(i0, ne10);

        float result = src0_row ? ggml_cuda_cast<float>(src0_row[size_t(i0)*s00]) : 0.0f;
        if constexpr (sizeof...(src1_ptrs) > 0) {
            result = (..., (result = bin_op(result, ggml_cuda_cast<float>(src1s[i_src1 + size_t(i10)*s10]))));
        } else {
            result = bin_op(result, ggml_cuda_cast<float>(src1[i_src1 + size_t(i10)*s10]));
        }

        dst_row[i0] = ggml_cuda_cast<dst_t>(result);

        // protect i0 from overflow
        if (ne0 - i0 <= s0) {
           break;
        }
    }
}

template <float (*bin_op)(const float, const float),
          typename src0_t,
          typename src1_t,
          typename dst_t,
          typename... src1_ptrs>
static __global__ void k_bin_bcast_unravel(const src0_t *         src0,
                                           const src1_t *         src1,
                                           dst_t *                dst,
                                           const uint3            ne0,
                                           const uint3            ne1,
                                           const uint3            ne2,
                                           const uint32_t         ne3,
                                           const uint3            prod_012,
                                           const uint3            prod_01,
                                           const uint3            ne10,
                                           const uint3            ne11,
                                           const uint3            ne12,
                                           const uint3            ne13,
                                         /*const int              s0,*/
                                           const uint32_t         s1,
                                           const uint32_t         s2,
                                           const uint32_t         s3,
                                           const uint32_t         s00,
                                           const uint32_t         s01,
                                           const uint32_t         s02,
                                           const uint32_t         s03,
                                           const uint32_t         s10,
                                           const uint32_t         s11,
                                           const uint32_t         s12,
                                           const uint32_t         s13,
                                           src1_ptrs... src1s) {
    const uint32_t i  = blockDim.x*blockIdx.x + threadIdx.x;

    const uint32_t i3 = fastdiv(i, prod_012);
    const uint32_t i2 = fastdiv(i - i3 * prod_012.z, prod_01);
    const uint32_t i1 = fastdiv(i - i3 * prod_012.z - i2 * prod_01.z, ne0);
    const uint32_t i0 = i - i3 * prod_012.z - i2 * prod_01.z - i1 * ne0.z;

    if (i0 >= ne0.z || i1 >= ne1.z || i2 >= ne2.z || i3 >= ne3) {
        return;
    }

    const uint32_t i11 = fastmodulo(i1, ne11);
    const uint32_t i12 = fastmodulo(i2, ne12);
    const uint32_t i13 = fastmodulo(i3, ne13);

    const size_t i_src0 = size_t( i3)*s03 + size_t( i2)*s02 + size_t( i1)*s01;
    const size_t i_src1 = size_t(i13)*s13 + size_t(i12)*s12 + size_t(i11)*s11;
    const size_t i_dst  = size_t( i3)*s3  + size_t( i2)*s2  + size_t( i1)*s1;

    const src0_t * src0_row = src0 ? (src0 + i_src0) : nullptr;
    dst_t * dst_row = dst + i_dst;

    const uint32_t i10 = fastmodulo(i0, ne10);

    ggml_cuda_pdl_sync();
    float result = src0_row ? ggml_cuda_cast<float>(src0_row[size_t(i0)*s00]) : 0.0f;
    if constexpr (sizeof...(src1_ptrs) > 0) {
        result = (..., (result = bin_op(result, ggml_cuda_cast<float>(src1s[i_src1 + size_t(i10)*s10]))));
    } else {
        result = bin_op(result, ggml_cuda_cast<float>(src1[i_src1 + size_t(i10)*s10]));
    }

    dst_row[i0] = ggml_cuda_cast<dst_t>(result);
}

template <float (*bin_op)(const float, const float), typename src0_t, typename src1_t, typename dst_t, size_t... I>
static void launch_bin_bcast_pack(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
                                  const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
                                  cudaStream_t stream, std::index_sequence<I...>) {
    GGML_TENSOR_BINARY_OP_LOCALS

    int nr0 = ne10 / ne0;
    int nr1 = ne11 / ne1;
    int nr2 = ne12 / ne2;
    int nr3 = ne13 / ne3;

    int nr[4] = { nr0, nr1, nr2, nr3 };

    int64_t cne[]  = { ne0, ne1, ne2, ne3 };
    int64_t cne0[] = { ne00, ne01, ne02, ne03 };
    int64_t cne1[] = { ne10, ne11, ne12, ne13 };

    size_t cnb[]  = { nb0, nb1, nb2, nb3 };
    size_t cnb0[] = { nb00, nb01, nb02, nb03 };
    size_t cnb1[] = { nb10, nb11, nb12, nb13 };

    auto collapse = [](int64_t cne[]) {
        cne[0] *= cne[1];
        cne[1] = cne[2];
        cne[2] = cne[3];
        cne[3] = 1;
    };

    auto collapse_nb = [](size_t cnb[], const int64_t cne[]) {
        cnb[1] *= cne[1];
        cnb[2] *= cne[2];
        cnb[3] *= cne[3];
    };

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && !ggml_is_permuted(src0) && !ggml_is_permuted(src1)) {
        for (int i = 0; i < 4; i++) {
            if (nr[i] != 1) {
                break;
            }
            if (i > 0) {
                collapse_nb(cnb, cne);
                collapse_nb(cnb0, cne0);
                collapse_nb(cnb1, cne1);
                collapse(cne);
                collapse(cne0);
                collapse(cne1);
            }
        }
    }

    {
        int64_t ne0 = cne[0];
        int64_t ne1 = cne[1];
        int64_t ne2 = cne[2];
        int64_t ne3 = cne[3];

        //int64_t ne00 = cne0[0]; GGML_UNUSED(ne00);
        //int64_t ne01 = cne0[1]; GGML_UNUSED(ne01);
        //int64_t ne02 = cne0[2]; GGML_UNUSED(ne02);
        //int64_t ne03 = cne0[3]; GGML_UNUSED(ne03);

        size_t nb0 = cnb[0];
        size_t nb1 = cnb[1];
        size_t nb2 = cnb[2];
        size_t nb3 = cnb[3];

        size_t nb00 = cnb0[0];
        size_t nb01 = cnb0[1];
        size_t nb02 = cnb0[2];
        size_t nb03 = cnb0[3];

        size_t nb10 = cnb1[0];
        size_t nb11 = cnb1[1];
        size_t nb12 = cnb1[2];
        size_t nb13 = cnb1[3];

      //size_t s0 = nb0 / sizeof(dst_t);
        size_t s1 = nb1 / sizeof(dst_t);
        size_t s2 = nb2 / sizeof(dst_t);
        size_t s3 = nb3 / sizeof(dst_t);

        size_t s10 = nb10 / sizeof(src1_t);
        size_t s11 = nb11 / sizeof(src1_t);
        size_t s12 = nb12 / sizeof(src1_t);
        size_t s13 = nb13 / sizeof(src1_t);

        size_t s00 = nb00 / sizeof(src0_t);
        size_t s01 = nb01 / sizeof(src0_t);
        size_t s02 = nb02 / sizeof(src0_t);
        size_t s03 = nb03 / sizeof(src0_t);

        GGML_ASSERT(ne0 <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(ne1 <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(ne2 <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(ne3 <= std::numeric_limits<uint32_t>::max());

      //GGML_ASSERT(s0  <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(s1  <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(s2  <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(s3  <= std::numeric_limits<uint32_t>::max());

        GGML_ASSERT(s00 <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(s01 <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(s02 <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(s03 <= std::numeric_limits<uint32_t>::max());

        GGML_ASSERT(s10 <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(s11 <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(s12 <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(s13 <= std::numeric_limits<uint32_t>::max());

        GGML_ASSERT(cne1[0] <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(cne1[1] <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(cne1[2] <= std::numeric_limits<uint32_t>::max());
        GGML_ASSERT(cne1[3] <= std::numeric_limits<uint32_t>::max());

        GGML_ASSERT(nb0 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb1 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb2 % sizeof(dst_t) == 0);
        GGML_ASSERT(nb3 % sizeof(dst_t) == 0);

        GGML_ASSERT(nb00 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb01 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb02 % sizeof(src0_t) == 0);
        GGML_ASSERT(nb03 % sizeof(src0_t) == 0);

        GGML_ASSERT(nb10 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb11 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb12 % sizeof(src1_t) == 0);
        GGML_ASSERT(nb13 % sizeof(src1_t) == 0);

        GGML_ASSERT(ne2 * ne3 <= std::numeric_limits<unsigned int>::max());

        const int block_size = 128;

        int64_t hne0 = std::max(ne0 / 2LL, 1LL);

        dim3 block_dims;
        block_dims.x = std::min<unsigned int>(hne0, block_size);
        block_dims.y = std::min<unsigned int>(ne1, block_size / block_dims.x);
        block_dims.z = std::min(std::min<unsigned int>(ne2 * ne3, block_size / block_dims.x / block_dims.y), 64U);

        dim3 block_nums((hne0 + block_dims.x - 1) / block_dims.x, (ne1 + block_dims.y - 1) / block_dims.y,
                        (ne2 * ne3 + block_dims.z - 1) / block_dims.z);

        const uint3 ne10 = init_fastdiv_values((uint32_t) cne1[0]);
        const uint3 ne11 = init_fastdiv_values((uint32_t) cne1[1]);
        const uint3 ne12 = init_fastdiv_values((uint32_t) cne1[2]);
        const uint3 ne13 = init_fastdiv_values((uint32_t) cne1[3]);

        if (block_nums.z > 65535 || block_nums.y > 65535) {
            int64_t     block_num   = (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;

            GGML_ASSERT(block_num              <= std::numeric_limits<uint32_t>::max());
            GGML_ASSERT(block_num * block_size <= std::numeric_limits<uint32_t>::max());
            GGML_ASSERT(ne0 * ne1              <= std::numeric_limits<uint32_t>::max());
            GGML_ASSERT(ne0 * ne1 * ne2        <= std::numeric_limits<uint32_t>::max());

            const uint3 prod_012    = init_fastdiv_values((uint32_t) (ne0 * ne1 * ne2));
            const uint3 prod_01     = init_fastdiv_values((uint32_t) (ne0 * ne1));
            const uint3 ne0_fastdiv = init_fastdiv_values((uint32_t) ne0);
            const uint3 ne1_fastdiv = init_fastdiv_values((uint32_t) ne1);
            const uint3 ne2_fastdiv = init_fastdiv_values((uint32_t) ne2);

            {
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params((dim3)block_num, block_size, 0, stream);
                ggml_cuda_kernel_launch(k_bin_bcast_unravel<bin_op, src0_t, src1_t, dst_t, type_for_index<const src1_t *, I>...>, launch_params,
                    src0_dd, src1_dd, dst_dd, ne0_fastdiv, ne1_fastdiv, ne2_fastdiv, ne3, prod_012, prod_01, ne10, ne11,
                    ne12, ne13,
                  /*s0,*/ s1,  s2,  s3,
                    s00, s01, s02, s03,
                    s10, s11, s12, s13, (const src1_t *) dst->src[I + 1]->data...);
            }
        } else {
            GGML_ASSERT(int64_t(block_nums.x) * block_dims.x <= std::numeric_limits<uint32_t>::max());
            GGML_ASSERT(int64_t(block_nums.y) * block_dims.y <= std::numeric_limits<uint32_t>::max());
            GGML_ASSERT(int64_t(block_nums.z) * block_dims.z <= std::numeric_limits<uint32_t>::max());

            const uint3 ne3_fastdiv = init_fastdiv_values((uint32_t) ne3);
            {
                const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(block_nums, block_dims, 0, stream);
                ggml_cuda_kernel_launch(k_bin_bcast<bin_op, src0_t, src1_t, dst_t, type_for_index<const src1_t *, I>...>, launch_params,
                    src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3_fastdiv, ne10, ne11, ne12, ne13,
                  /*s0,*/ s1, s2,  s3,
                    s00, s01, s02, s03,
                    s10, s11, s12, s13, (const src1_t *) dst->src[I + 1]->data...);
            }
        }
    }
}

template <typename T>
static __global__ void k_repeat_back(
    const T * __restrict__ src, T * __restrict__ dst, const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const size_t s00, const size_t s01, const size_t s02, const size_t s03,
    const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3) {

    const int64_t tid0  = int64_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const int64_t tid1  = int64_t(blockIdx.y)*blockDim.y + threadIdx.y;
    const int64_t tid23 = int64_t(blockIdx.z)*blockDim.z + threadIdx.z;
    const int64_t tid2  = tid23 % ne2;
    const int64_t tid3  = tid23 / ne2;

    if (tid0 >= ne0) {
        return;
    }

    T sum = 0;
    ggml_cuda_pdl_sync();
    for (int64_t i3 = tid3; i3 < ne03; i3 += ne3) {
        for (int64_t i2 = tid2; i2 < ne02; i2 += ne2) {
            for (int64_t i1 = tid1; i1 < ne01; i1 += ne1) {
                for (int64_t i0 = tid0; i0 < ne00; i0 += ne0) {
                    sum += src[i3*s03 + i2*s02 + i1*s01 + i0*s00];
                }
            }
        }
    }
    dst[tid3*ne2*ne1*ne0 + tid2*ne1*ne0 + tid1*ne0 + tid0] = sum;
}

template <float (*bin_op)(const float, const float), int n_fuse = 1>
struct bin_bcast_cuda {
    template<typename src0_t, typename src1_t, typename dst_t>
    void operator()(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst,
            const src0_t * src0_dd, const src1_t * src1_dd, dst_t * dst_dd,
            cudaStream_t stream) {
        launch_bin_bcast_pack<bin_op, src0_t, src1_t, dst_t>(
            src0, src1, dst, src0_dd, src1_dd, dst_dd, stream, std::make_index_sequence<n_fuse>{});
    }
};

template <typename T>
static void repeat_back_cuda(
    const T * src, T * dst, const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
    const size_t s00, const size_t s01, const size_t s02, const size_t s03,
    const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {

    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums((ne0 + WARP_SIZE - 1) / WARP_SIZE, ne1, ne2*ne3);
    k_repeat_back<T><<<block_nums, block_dims, 0, stream>>>
        (src, dst, ne00, ne01, ne02, ne03, s00, s01, s02, s03, ne0, ne1, ne2, ne3);
}

template<class op>
static void ggml_cuda_op_bin_bcast(
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const void * src0_dd, const void * src1_dd, void * dst_dd, cudaStream_t stream) {

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_BF16);

    if (src0->type == GGML_TYPE_F32 && src1->type != GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const float *)src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        op()(src0, src1, dst, (const nv_bfloat16 *) src0_dd, (const nv_bfloat16 *) src1_dd, (nv_bfloat16 *) dst_dd, stream);
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        op()(src0, src1, dst, (const nv_bfloat16 *) src0_dd, (const float *) src1_dd, (nv_bfloat16 *) dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()(src0, src1, dst, (const half *) src0_dd, (const half *)src1_dd, (half *) dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()(src0, src1, dst, (const half *) src0_dd, (const float *)src1_dd, (half *) dst_dd, stream);
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        op()(src0, src1, dst, (const half *) src0_dd, (const float *)src1_dd, (float *)dst_dd, stream);
    } else {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__,
            ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_repeat, 0>>(dst, dst->src[0], dst, nullptr, dst->src[0]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_add>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_sub>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_mul>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_op_bin_bcast<bin_bcast_cuda<op_div>>(dst->src[0], dst->src[1], dst, dst->src[0]->data, dst->src[1]->data, dst->data, ctx.stream());
}

template <float (*op)(const float, const float), int n_fuse>
static void ggml_cuda_op_fused_binbcast_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src0->type == GGML_TYPE_F32 && src1->type != GGML_TYPE_BF16 && dst->type == GGML_TYPE_F32) {
        launch_bin_bcast_pack<op, float, float, float>(src0, src1, dst,
            (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast_pack<op, nv_bfloat16, nv_bfloat16, nv_bfloat16>(src0, src1, dst,
            (const nv_bfloat16 *) src0->data, (const nv_bfloat16 *) src1->data, (nv_bfloat16 *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast_pack<op, nv_bfloat16, float, nv_bfloat16>(src0, src1, dst,
            (const nv_bfloat16 *) src0->data, (const float *) src1->data, (nv_bfloat16 *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        launch_bin_bcast_pack<op, half, half, half>(src0, src1, dst,
            (const half *) src0->data, (const half *) src1->data, (half *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else if (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        launch_bin_bcast_pack<op, half, float, half>(src0, src1, dst,
            (const half *) src0->data, (const float *) src1->data, (half *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else if (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F32) {
        launch_bin_bcast_pack<op, half, float, float>(src0, src1, dst,
            (const half *) src0->data, (const float *) src1->data, (float *) dst->data,
            stream, std::make_index_sequence<n_fuse>{});
    } else {
        fprintf(stderr,
                "%s: unsupported types for fusion: dst: %s, src0: %s, src1: %s\n",
                __func__, ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}


void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse) {
    GGML_ASSERT(2 <= n_fuse && n_fuse <= 8);

    switch (n_fuse) {
        case 2:
            ggml_cuda_op_fused_binbcast_impl<op_add, 2>(ctx, dst);
            break;
        case 3:
            ggml_cuda_op_fused_binbcast_impl<op_add, 3>(ctx, dst);
            break;
        case 4:
            ggml_cuda_op_fused_binbcast_impl<op_add, 4>(ctx, dst);
            break;
        case 5:
            ggml_cuda_op_fused_binbcast_impl<op_add, 5>(ctx, dst);
            break;
        case 6:
            ggml_cuda_op_fused_binbcast_impl<op_add, 6>(ctx, dst);
            break;
        case 7:
            ggml_cuda_op_fused_binbcast_impl<op_add, 7>(ctx, dst);
            break;
        case 8:
            ggml_cuda_op_fused_binbcast_impl<op_add, 8>(ctx, dst);
            break;
        default:
            GGML_ASSERT(false && "Unsupported n_fuse value");
    }
}

void ggml_cuda_op_fused_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse) {
    GGML_ASSERT(2 <= n_fuse && n_fuse <= 8);

    switch (n_fuse) {
        case 2:
            ggml_cuda_op_fused_binbcast_impl<op_mul, 2>(ctx, dst);
            break;
        case 3:
            ggml_cuda_op_fused_binbcast_impl<op_mul, 3>(ctx, dst);
            break;
        case 4:
            ggml_cuda_op_fused_binbcast_impl<op_mul, 4>(ctx, dst);
            break;
        case 5:
            ggml_cuda_op_fused_binbcast_impl<op_mul, 5>(ctx, dst);
            break;
        case 6:
            ggml_cuda_op_fused_binbcast_impl<op_mul, 6>(ctx, dst);
            break;
        case 7:
            ggml_cuda_op_fused_binbcast_impl<op_mul, 7>(ctx, dst);
            break;
        case 8:
            ggml_cuda_op_fused_binbcast_impl<op_mul, 8>(ctx, dst);
            break;
        default:
            GGML_ASSERT(false && "Unsupported n_fuse value");
    }
}

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_can_repeat(dst, src0));

    cudaStream_t stream = ctx.stream();

    GGML_TENSOR_UNARY_OP_LOCALS;

    GGML_ASSERT(ne2*ne3 <= (1 << 15));

    const size_t ts = ggml_type_size(src0->type);
    const size_t s00 = nb00 / ts;
    const size_t s01 = nb01 / ts;
    const size_t s02 = nb02 / ts;
    const size_t s03 = nb03 / ts;

    switch (dst->type) {
        case GGML_TYPE_F32: {
            const float * src0_d = (const float *) src0->data;
            float       * dst_d  = (float       *) dst->data;
            repeat_back_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s00, s01, s02, s03, ne0, ne1, ne2, ne3, stream);
        } break;
        default: {
            GGML_ASSERT(false);
        } break;
    }
}

// ===== [gfx906 P6] adjacent elementwise chain: 2..8 nodes -> 1 launch =====
// Why: every elementwise node in this model is one launch of ~4.2 us GPU time no matter
// how small the tensor is (measured 581 elementwise launches/token, 2.44 ms/token in
// experiment-2026-10-18/current-trace). Chains of 2..8 single-consumer nodes are already
// adjacent in graph order, so evaluating the run in one kernel removes n-1 launches.
//
// Math: each step applies exactly the expression its solo kernel uses (the same device
// functions from unary.cuh, the same a+b / a-b / a*b / a/b, scale*x+bias, fminf/fmaxf
// clamp order). No reduction order changes, so accepted chains are bit-identical.
//
// Safety: the caller (gfx906_elem_chain_try) requires every node but the last to have
// exactly one use inside the run (ggml_can_fuse_subgraph), all nodes share one shape,
// every operand is F32 row-contiguous with dims {1, ne}, and no live read overlaps the
// chain output except an exact element-wise in-place alias.

static int gfx906_ec_code(const ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_ADD:   return GFX906_EC_ADD;
        case GGML_OP_SUB:   return GFX906_EC_SUB;
        case GGML_OP_MUL:   return GFX906_EC_MUL;
        case GGML_OP_DIV:   return GFX906_EC_DIV;
        case GGML_OP_SCALE: return GFX906_EC_SCALE;
        case GGML_OP_CLAMP: return GFX906_EC_CLAMP;
        case GGML_OP_SQRT:  return GFX906_EC_SQRT;
        case GGML_OP_SQR:   return GFX906_EC_SQR;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(t)) {
                case GGML_UNARY_OP_ABS:      return GFX906_EC_ABS;
                case GGML_UNARY_OP_NEG:      return GFX906_EC_NEG;
                case GGML_UNARY_OP_TANH:     return GFX906_EC_TANH;
                case GGML_UNARY_OP_SIGMOID:  return GFX906_EC_SIGMOID;
                case GGML_UNARY_OP_SILU:     return GFX906_EC_SILU;
                case GGML_UNARY_OP_SOFTPLUS: return GFX906_EC_SOFTPLUS;
                case GGML_UNARY_OP_EXP:      return GFX906_EC_EXP;
                case GGML_UNARY_OP_RELU:     return GFX906_EC_RELU;
                case GGML_UNARY_OP_GELU:     return GFX906_EC_GELU;
                case GGML_UNARY_OP_GELU_QUICK: return GFX906_EC_GELU_QUICK;
                case GGML_UNARY_OP_GELU_ERF:   return GFX906_EC_GELU_ERF;
                default: return -1;
            }
        default: return -1;
    }
}

// operand read by a step: F32, dims 0/1 either 1 (broadcast) or the chain's, no dims 2/3,
// and the contiguous dim must really be contiguous when it is not broadcast
static bool gfx906_ec_read_ok(const ggml_tensor * t, const ggml_tensor * out) {
    if (t == nullptr || t->data == nullptr || t->type != GGML_TYPE_F32) {
        return false;
    }
    if (t->ne[2] != 1 || t->ne[3] != 1) {
        return false;
    }
    if (t->ne[0] != out->ne[0] && t->ne[0] != 1) {
        return false;
    }
    if (t->ne[0] == out->ne[0] && t->nb[0] != sizeof(float)) {
        return false;
    }
    return t->ne[1] == out->ne[1] || t->ne[1] == 1;
}

static bool gfx906_ec_overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * a1 = a0 + ggml_nbytes(a);
    const char * b0 = (const char *) b->data;
    const char * b1 = b0 + ggml_nbytes(b);
    return a0 < b1 && b0 < a1;
}

// a live read may only touch the chain output as an exact element-wise in-place alias
static bool gfx906_ec_alias_ok(const ggml_tensor * t, const ggml_tensor * dst) {
    if (t == nullptr || t->data == nullptr || dst->data == nullptr) {
        return false;
    }
    if (!gfx906_ec_overlaps(t, dst)) {
        return true;
    }
    return t->data == dst->data && ggml_are_same_shape(t, dst);
}

static void gfx906_ec_fill_operand(gfx906_elem_chain_step & st, const ggml_tensor * op) {
    st.p  = (const float *) op->data;
    st.s0 = op->ne[0] == 1 ? 0 : 1;
    st.s1 = op->ne[1] == 1 ? 0 : (int64_t) (op->nb[1] / sizeof(float));
}

static void gfx906_ec_miss(const ggml_cgraph * cgraph, int i, int reason) {
    static const char * v = getenv("GFX906_ELEM_CHAIN_MISS");
    if (v == nullptr || reason == 2) {   // reason 2 = "not an elementwise op": noise
        return;
    }
    static int n = 0;
    if (n++ >= 240) {
        return;
    }
    const ggml_tensor * h = cgraph->nodes[i];
    const ggml_tensor * a = h->src[0];
    const ggml_tensor * b = h->src[1];
    fprintf(stderr, "[ec-miss] i=%d reason=%d op=%s ne=[%lld,%lld] src0=%s(%s) ne=[%lld,%lld,%lld,%lld] src1=%s(%s) ne=[%lld,%lld,%lld,%lld]\n",
            i, reason, ggml_op_desc(h), (long long) h->ne[0], (long long) h->ne[1],
            a ? ggml_op_desc(a) : "-", a ? ggml_type_name(a->type) : "-",
            a ? (long long) a->ne[0] : 0, a ? (long long) a->ne[1] : 0, a ? (long long) a->ne[2] : 0, a ? (long long) a->ne[3] : 0,
            b ? ggml_op_desc(b) : "-", b ? ggml_type_name(b->type) : "-",
            b ? (long long) b->ne[0] : 0, b ? (long long) b->ne[1] : 0, b ? (long long) b->ne[2] : 0, b ? (long long) b->ne[3] : 0);
}

static bool gfx906_elem_chain_match(const ggml_cgraph * cgraph, int i, gfx906_elem_chain_plan & plan) {
    plan.n  = 0;
    plan.in = nullptr;
    if (i + 1 >= cgraph->n_nodes) {
        gfx906_ec_miss(cgraph, i, 1);
        return false;
    }
    const ggml_tensor * head = cgraph->nodes[i];
    if (gfx906_ec_code(head) < 0 || head->type != GGML_TYPE_F32 ||
            head->ne[2] != 1 || head->ne[3] != 1) {
        gfx906_ec_miss(cgraph, i, 2);
        return false;
    }

    const ggml_tensor * prev = nullptr;
    for (int k = 0; k < GFX906_EC_MAX_STEPS && i + k < cgraph->n_nodes; ++k) {
        const ggml_tensor * node = cgraph->nodes[i + k];
        const int code = gfx906_ec_code(node);
        if (code < 0 || node->type != GGML_TYPE_F32 || node->ne[2] != 1 || node->ne[3] != 1) {
            break;
        }
        if (!ggml_are_same_shape(node, head)) {
            if (k == 0) gfx906_ec_miss(cgraph, i, 3);
            break;
        }

        gfx906_elem_chain_step st = {};
        st.op = code;

        if (code <= GFX906_EC_DIV) {          // binary op
            const ggml_tensor * a = node->src[0];
            const ggml_tensor * b = node->src[1];
            if (k == 0) {
                // head: the stream is src[0] (that is exactly the solo op's operand order,
                // including SUB/DIV) and src[1] is the broadcast/paired operand
                const ggml_tensor * in = a;
                const ggml_tensor * op = b;
                if (!gfx906_ec_read_ok(in, head)) {
                    if (k == 0) gfx906_ec_miss(cgraph, i, 4);
                    break;
                }
                plan.in = in;
                if (op != nullptr) {
                    if (!gfx906_ec_read_ok(op, head)) {
                        if (k == 0) gfx906_ec_miss(cgraph, i, 6);
                        break;
                    }
                    gfx906_ec_fill_operand(st, op);
                }
            } else {
                const ggml_tensor * op = nullptr;
                if (a == prev) {
                    op = b;
                } else if (b == prev && (code == GFX906_EC_ADD || code == GFX906_EC_MUL)) {
                    op = a;
                } else {
                    if (k == 0) gfx906_ec_miss(cgraph, i, 5);
                    break;
                }
                if (op != nullptr) {
                    if (!gfx906_ec_read_ok(op, head)) {
                        break;
                    }
                    gfx906_ec_fill_operand(st, op);
                }
            }
        } else {                              // unary-shaped op: value comes from src[0]
            const ggml_tensor * x = node->src[0];
            if (k == 0) {
                if (!gfx906_ec_read_ok(x, head) || !ggml_are_same_shape(x, head)) {
                    gfx906_ec_miss(cgraph, i, 7);
                    break;
                }
                plan.in = x;
            } else if (x != prev) {
                break;
            }
            if (code == GFX906_EC_SCALE) {
                st.a = ggml_get_op_params_f32(node, 0);
                memcpy(&st.b, (const float *) node->op_params + 1, sizeof(float));
            } else if (code == GFX906_EC_CLAMP) {
                st.a = ggml_get_op_params_f32(node, 0);
                st.b = ggml_get_op_params_f32(node, 1);
            }
        }

        plan.steps[plan.n] = st;
        plan.nodes[plan.n] = node;
        ++plan.n;
        prev = node;
    }

    return plan.n >= 2;
}

// every live read must stay out of the chain output's storage (except an exact in-place alias)
static bool gfx906_elem_chain_live_ok(const gfx906_elem_chain_plan & plan) {
    const ggml_tensor * dst = plan.nodes[plan.n - 1];
    if (!gfx906_ec_alias_ok(plan.in, dst)) {
        return false;
    }
    for (int k = 0; k < plan.n; ++k) {
        if (plan.steps[k].p == nullptr) {
            continue;
        }
        // steps only carry the operand pointer; recover the tensor from the node's srcs
        const ggml_tensor * node = plan.nodes[k];
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = node->src[s];
            if (src != nullptr && (const float *) src->data == plan.steps[k].p) {
                if (!gfx906_ec_alias_ok(src, dst)) {
                    return false;
                }
            }
        }
    }
    return true;
}

static __device__ __forceinline__ float gfx906_ec_apply(const int op, const float v, const float x,
        const float a, const float b) {
    switch (op) {
        case GFX906_EC_ADD:      return v + x;
        case GFX906_EC_SUB:      return v - x;
        case GFX906_EC_MUL:      return v * x;
        case GFX906_EC_DIV:      return v / x;
        case GFX906_EC_SCALE:    return a * v + b;
        case GFX906_EC_CLAMP:    return fminf(fmaxf(v, a), b);
        case GFX906_EC_SQRT:     return sqrtf(v);
        case GFX906_EC_SQR:      return v * v;
        case GFX906_EC_ABS:      return fabsf(v);
        case GFX906_EC_NEG:      return -v;
        case GFX906_EC_TANH:     return tanhf(v);
        case GFX906_EC_SIGMOID:  return 1.0f / (1.0f + expf(-v));
        case GFX906_EC_SILU:     return ggml_cuda_op_silu_single(v);
        case GFX906_EC_SOFTPLUS: return (v > 20.0f) ? v : logf(1.0f + expf(v));
        case GFX906_EC_EXP:      return expf(v);
        case GFX906_EC_RELU:     return fmaxf(v, 0.0f);
        case GFX906_EC_GELU:     return ggml_cuda_op_gelu_single(v);
        case GFX906_EC_GELU_QUICK: return v * (1.0f / (1.0f + expf(-1.702f * v)));
        case GFX906_EC_GELU_ERF:   return 0.5f*v*(1.0f + erff(v*0.70710678118654752440084436210484f));
    }
    return v;
}

struct gfx906_elem_chain_args {
    const float *              in;
    int64_t                    in_s0, in_s1;
    float *                    out;
    int                        ne0, ne1, n;
    gfx906_elem_chain_step     st[GFX906_EC_MAX_STEPS];
};

static __global__ void k_gfx906_elem_chain(const gfx906_elem_chain_args args) {
    const int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    const int i1 = blockIdx.y;
    if (i0 >= args.ne0 || i1 >= args.ne1) {
        return;
    }
    float v = args.in[(int64_t) i0*args.in_s0 + (int64_t) i1*args.in_s1];
    for (int k = 0; k < args.n; ++k) {
        const gfx906_elem_chain_step & st = args.st[k];
        const float x = st.p != nullptr ? st.p[(int64_t) i0*st.s0 + (int64_t) i1*st.s1] : 0.0f;
        v = gfx906_ec_apply(st.op, v, x, st.a, st.b);
    }
    args.out[(int64_t) i1*args.ne0 + i0] = v;
}

static void gfx906_elem_chain_launch(ggml_backend_cuda_context & ctx, const gfx906_elem_chain_plan & plan) {
    const ggml_tensor * dst = plan.nodes[plan.n - 1];

    gfx906_elem_chain_args args = {};
    args.in    = (const float *) plan.in->data;
    args.in_s0 = plan.in->ne[0] == 1 ? 0 : 1;
    args.in_s1 = plan.in->ne[1] == 1 ? 0 : (int64_t) (plan.in->nb[1] / sizeof(float));
    args.out   = (float *) dst->data;
    args.ne0   = (int) dst->ne[0];
    args.ne1   = (int) dst->ne[1];
    args.n     = plan.n;
    for (int k = 0; k < plan.n; ++k) {
        args.st[k] = plan.steps[k];
    }

    const int block = 256;
    const dim3 grid((args.ne0 + block - 1) / block, args.ne1);
    k_gfx906_elem_chain<<<grid, block, 0, ctx.stream()>>>(args);
    CUDA_CHECK(cudaGetLastError());
}

// per-run tally so a silent mismatch (pattern miss, veto) is visible
static long g_ec_hits = 0, g_ec_nodes = 0, g_ec_chain_hist[GFX906_EC_MAX_STEPS + 1] = {};

int gfx906_elem_chain_try(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i, bool dry) {
    gfx906_elem_chain_plan plan;
    if (!gfx906_elem_chain_match(cgraph, i, plan)) {
        return 0;
    }
    enum ggml_op ops[GFX906_EC_MAX_STEPS];
    for (int k = 0; k < plan.n; ++k) {
        ops[k] = plan.nodes[k]->op;
    }
    const int out_idx = i + plan.n - 1;
    if (!ggml_can_fuse_subgraph(cgraph, i, plan.n, ops, &out_idx, 1)) {
        gfx906_ec_miss(cgraph, i, 8);
        return 0;
    }
    if (!gfx906_elem_chain_live_ok(plan)) {
        gfx906_ec_miss(cgraph, i, 9);
        return 0;
    }

    ++g_ec_hits;
    g_ec_nodes += plan.n;
    ++g_ec_chain_hist[plan.n];
    static const bool dbg = getenv("GFX906_ELEM_CHAIN_DEBUG") != nullptr;
    if (dbg && g_ec_hits <= 400) {
        fprintf(stderr, "[elem-chain] n=%d ne=[%lld,%lld] op0=%s -> %s (saved %d launches)\n",
                plan.n, (long long) cgraph->nodes[i]->ne[0], (long long) cgraph->nodes[i]->ne[1],
                ggml_op_desc(cgraph->nodes[i]), cgraph->nodes[out_idx]->name, plan.n - 1);
    }

    if (dbg && (g_ec_hits % 256) == 0) {
        fprintf(stderr, "[elem-chain] cumulative hits=%ld nodes=%ld launches_saved=%ld\n",
                g_ec_hits, g_ec_nodes, g_ec_nodes - g_ec_hits);
    }

    if (!dry) {
        gfx906_elem_chain_launch(ctx, plan);
    }
    return plan.n;
}
