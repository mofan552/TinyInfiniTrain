#include "cublas_v2.h"
#include "glog/logging.h"
#include <algorithm>
#include <cub/block/block_reduce.cuh>
#include <numeric>
#include <tuple>
#include <vector>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cuda {

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess) {                                                                                   \
            LOG(FATAL) << "CUDA Error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__;       \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        cublasStatus_t status = call;                                                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                         \
            LOG(FATAL) << "CUBLAS Error: " << cublasGetStatusString(status) << " at " << __FILE__ << ":" << __LINE__;  \
        }                                                                                                              \
    } while (0)

namespace {
// 矩阵乘的形状信息：最后两维参与乘法，其余前置维度折叠成 batch。
struct MatmulDims {
    int64_t m = 0;        // input 的倒数第二维
    int64_t k = 0;        // 收缩维
    int64_t n = 0;        // other 的最后一维
    int64_t input_bs = 1; // input 折叠后的 batch 数
    int64_t other_bs = 1; // other 折叠后的 batch 数
    int64_t bs = 1;       // 广播后的 batch 数
};

MatmulDims ComputeMatmulDims(const std::vector<int64_t> &input_dims, const std::vector<int64_t> &other_dims) {
    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);

    MatmulDims d;
    d.m = input_dims[input_dims.size() - 2];
    d.k = *input_dims.rbegin();
    CHECK_EQ(other_dims[other_dims.size() - 2], d.k)
        << "matmul shape mismatch: input's last dim must equal other's second-to-last dim";
    d.n = *other_dims.rbegin();

    d.input_bs = std::accumulate(input_dims.begin(), input_dims.end() - 2, int64_t{1}, std::multiplies<int64_t>{});
    d.other_bs = std::accumulate(other_dims.begin(), other_dims.end() - 2, int64_t{1}, std::multiplies<int64_t>{});
    d.bs = std::max(d.input_bs, d.other_bs);
    CHECK(d.input_bs == d.bs || d.input_bs == 1) << "unsupported batch broadcast on input";
    CHECK(d.other_bs == d.bs || d.other_bs == 1) << "unsupported batch broadcast on other";
    return d;
}
} // namespace

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CUDA上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================

    /*
      input: (*, m, k) @ other: (*, k, n) -> output: (*, m, n)，数据均为行主序。
      cuBLAS 采用列主序，行主序矩阵 X(a, b) 在列主序下即 X^T(b, a)，因此
        output = input @ other   <==>   output^T = other^T @ input^T
      直接把 other 当作第一个操作数、input 当作第二个操作数传给 cublas 即可，
      无需任何显式转置（与本文件中 LinearForward 的处理方式一致）。
      广播的一侧把 stride 置 0，让所有 batch 复用同一块显存。
    */
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto d = ComputeMatmulDims(input_dims, other_dims);

    auto output_dims = d.input_bs >= d.other_bs ? std::vector<int64_t>(input_dims.begin(), input_dims.end() - 2)
                                                : std::vector<int64_t>(other_dims.begin(), other_dims.end() - 2);
    output_dims.push_back(d.m);
    output_dims.push_back(d.n);
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    const auto *input_ptr = static_cast<const float *>(input->DataPtr());
    const auto *other_ptr = static_cast<const float *>(other->DataPtr());
    auto *output_ptr = static_cast<float *>(output->DataPtr());

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // 强制严格 FP32：禁用 TF32 张量核以及 split-k 等会改变累加顺序的快速路径，
    // 保证结果与参考实现在数值上尽可能一致
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
    for (int64_t b = 0; b < d.bs; ++b) {
        // C^T[n, m] = other^T[n, k] * input^T[k, m]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d.n, d.m, d.k, &alpha,
                                 other_ptr + (d.other_bs == 1 ? 0 : b * d.k * d.n), d.n,
                                 input_ptr + (d.input_bs == 1 ? 0 : b * d.m * d.k), d.k, &beta,
                                 output_ptr + b * d.m * d.n, d.n));
    }
    CUBLAS_CHECK(cublasDestroy(handle));

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CUDA上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================

    /*
      grad_input[*, m, k] = grad_output[*, m, n] @ other[*, k, n]^T
      grad_other[*, k, n] = input[*, m, k]^T @ grad_output[*, m, n]
      同样按列主序做等价换位：
        grad_input^T[k, m] = other[k, n](列主序视作 other^T) ... 见下方各次调用的注释。
      若某一侧发生 batch 广播，其梯度需在 batch 维上累加，此时退化为逐 batch
      调用并令 beta = 1 累积，避免 strided batched 写同一块显存产生竞争。
    */
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto d = ComputeMatmulDims(input_dims, other_dims);
    CHECK_EQ(grad_output->NumElements(), d.bs * d.m * d.n);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32, other->GetDevice());

    const auto *input_ptr = static_cast<const float *>(input->DataPtr());
    const auto *other_ptr = static_cast<const float *>(other->DataPtr());
    const auto *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    auto *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());
    auto *grad_other_ptr = static_cast<float *>(grad_other->DataPtr());

    const float alpha = 1.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    // 广播的一侧需要在 batch 维累加，先清零再用 beta = 1 累积；未广播时直接覆盖写入
    const bool accumulate_grad_input = d.input_bs != d.bs;
    const bool accumulate_grad_other = d.other_bs != d.bs;
    if (accumulate_grad_input) {
        CUDA_CHECK(cudaMemset(grad_input_ptr, 0, grad_input->SizeInBytes()));
    }
    if (accumulate_grad_other) {
        CUDA_CHECK(cudaMemset(grad_other_ptr, 0, grad_other->SizeInBytes()));
    }
    const float beta_input = accumulate_grad_input ? 1.0f : 0.0f;
    const float beta_other = accumulate_grad_other ? 1.0f : 0.0f;

    for (int64_t b = 0; b < d.bs; ++b) {
        const int64_t input_b = d.input_bs == 1 ? 0 : b;
        const int64_t other_b = d.other_bs == 1 ? 0 : b;

        // ---- grad_input = grad_output @ other^T ----
        // 列主序：C[k, m] = op(other)[k, n] * op(grad_output)[n, m]
        // other 行主序 (k, n) -> 列主序 (n, k)，需转置得到 (k, n)，故 opA = T，lda = n
        // grad_output 行主序 (m, n) -> 列主序 (n, m)，直接可用，opB = N，ldb = n
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, d.k, d.m, d.n, &alpha,
                                 other_ptr + other_b * d.k * d.n, d.n, grad_output_ptr + b * d.m * d.n, d.n,
                                 &beta_input, grad_input_ptr + input_b * d.m * d.k, d.k));

        // ---- grad_other = input^T @ grad_output ----
        // 列主序：C[n, k] = op(grad_output)[n, m] * op(input)[m, k]
        // grad_output 行主序 (m, n) -> 列主序 (n, m)，opA = N，lda = n
        // input 行主序 (m, k) -> 列主序 (k, m)，需转置得到 (m, k)，故 opB = T，ldb = k
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, d.n, d.k, d.m, &alpha,
                                 grad_output_ptr + b * d.m * d.n, d.n, input_ptr + input_b * d.m * d.k, d.k,
                                 &beta_other, grad_other_ptr + other_b * d.k * d.n, d.n));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    return {grad_input, grad_other};
}

__global__ void BiasCopyKernel(float *output, const float *bias, int bs, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bs * out_features) {
        return;
    }
    int j = idx % out_features;
    output[idx] = bias[j];
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {

    /*
        !transpose: output = input * weight + bias
        output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]

        transpose:  output = input * weight^T + bias
        output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);

    // As for cublas:
    // C = alpha * op(B) * op(A) + beta * C
    // Dimensions:
    //   input:  (bs, in_features)
    //   weight: (in_features, out_features) or (out_features, in_features) if transposed
    //   output: (bs, out_features)
    const int64_t out_features = weight_dims[transpose ? 0 : 1];

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    if (bias) {
        CHECK_EQ(bias->Dims().size(), 1);
        CHECK_EQ(bias->Dims()[0], out_features);
        int threads_per_block = 256;
        int num_blocks = (bs * out_features + threads_per_block - 1) / threads_per_block;
        BiasCopyKernel<<<num_blocks, threads_per_block>>>(
            static_cast<float *>(output->DataPtr()), static_cast<const float *>(bias->DataPtr()), bs, out_features);
    } else {
        output->Fill<float>(0.0f);
    }

    const float alpha = 1.0f;
    const float beta = 1.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    if (transpose) {
        // weight is [out_features, in_features] here

        // output = input * weight.T --> output.T = weight * input.T
        // C = output.T[out_features, bs]
        // A = weight.T[in_features, out_features]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, bs, in_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(output->DataPtr()), out_features));
    } else {
        // output = input * weight --> output.T =  weight.T * input.T
        // C = output.T[out_features, bs]
        // A = weight.T[out_features, in_features]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out_features, bs, in_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), out_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(output->DataPtr()), out_features));
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    return output;
}

template <int BLOCK_SIZE>
__global__ void ReduceColumnsKernel(const float *__restrict__ input, float *__restrict__ output, int num_rows,
                                    int num_cols) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int row = blockIdx.x;
    float sum = 0.0f;

    for (int col = threadIdx.x; col < num_cols; col += blockDim.x) { sum += input[row * num_cols + col]; }

    float reduced = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        output[row] = reduced;
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, grad_output->GetDevice());
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32, grad_output->GetDevice());
    grad_input->Fill<float>(0.0f);
    grad_weight->Fill<float>(0.0f);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32,
                                             grad_output->GetDevice());
        grad_bias->Fill<float>(0.0f);
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    if (transpose) {
        // weight is [out_features, in_features] here

        // d_input = d_output * weight --> d_input.T = weight.T * d_output.T
        // C = d_input.T[in_features, bs]
        // A = weight.T[in_features, out_features]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), in_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = d_output.T * input --> d_weight.T = input.T * d_output
        // C = d_weight.T[in_features, out_features]
        // A = input.T[in_features, bs]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, in_features, out_features, bs, &alpha,
                                 static_cast<const float *>(input->DataPtr()), in_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_weight->DataPtr()), in_features));
    } else {
        // weight is [in_features, out_features] here

        // d_input = d_output * weight.T --> d_input.T = weight * d_output.T
        // C = d_input.T[in_features, bs]
        // A = weight.T[out_features, in_features]
        // B = d_output.T[out_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, in_features, bs, out_features, &alpha,
                                 static_cast<const float *>(weight->DataPtr()), out_features,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features, &beta,
                                 static_cast<float *>(grad_input->DataPtr()), in_features));

        // d_weight = input.T * d_output --> d_weight.T = d_output.T * input
        // C = d_weight.T[out_features, in_features]
        // A = d_output.T[out_features, bs]
        // B = input.T[in_features, bs]
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, out_features, in_features, bs, &alpha,
                                 static_cast<const float *>(grad_output->DataPtr()), out_features,
                                 static_cast<const float *>(input->DataPtr()), in_features, &beta,
                                 static_cast<float *>(grad_weight->DataPtr()), out_features));
    }

    // d_bias = \sum_i(i=0, bs-1) d_output[i]
    if (bias) {
        constexpr int BLOCK_SIZE = 256;
        int threads_per_block = BLOCK_SIZE;
        int num_blocks = out_features;
        ReduceColumnsKernel<BLOCK_SIZE>
            <<<num_blocks, threads_per_block>>>(static_cast<const float *>(grad_output->DataPtr()),
                                                static_cast<float *>(grad_bias->DataPtr()), out_features, bs);
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cuda

#define REGISTER_CUDA_LINEAR_KERNEL(kernel_name)                                                                       \
    REGISTER_KERNEL(infini_train::DeviceType::kCUDA, kernel_name, infini_train::kernels::cuda::kernel_name)

REGISTER_CUDA_LINEAR_KERNEL(MatmulForward)
REGISTER_CUDA_LINEAR_KERNEL(MatmulBackward)
REGISTER_CUDA_LINEAR_KERNEL(LinearForward)
REGISTER_CUDA_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CUDA_LINEAR_KERNEL
