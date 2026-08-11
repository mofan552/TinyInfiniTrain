#include <algorithm>
#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
namespace {
using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixMap = Eigen::Map<RowMajorMatrix>;
using ConstMatrixMap = Eigen::Map<const RowMajorMatrix>;

// 矩阵乘的形状信息：最后两维参与乘法，其余前置维度折叠成 batch。
struct MatmulDims {
    int64_t m = 0;         // input 的倒数第二维
    int64_t k = 0;         // 收缩维
    int64_t n = 0;         // other 的最后一维
    int64_t input_bs = 1;  // input 折叠后的 batch 数
    int64_t other_bs = 1;  // other 折叠后的 batch 数
    int64_t bs = 1;        // 广播后的 batch 数
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
    // 只支持其中一侧 batch 为 1 的广播（含二维矩阵与 batch 矩阵相乘的情形）
    CHECK(d.input_bs == d.bs || d.input_bs == 1) << "unsupported batch broadcast on input";
    CHECK(d.other_bs == d.bs || d.other_bs == 1) << "unsupported batch broadcast on other";
    return d;
}
} // namespace

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================

    /*
      input:  (*, m, k)
      other:  (*, k, n)
      output: (*, m, n)
      注意不能直接用 Tensor::EigenMatrix()，它把除最后一维外的所有维度折叠成行，
      对 batch 矩阵乘会得到错误的形状，这里按 batch 逐个建立 Eigen::Map。
    */
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto d = ComputeMatmulDims(input_dims, other_dims);

    // 输出的前置维度取 batch 更多的一侧
    auto output_dims = d.input_bs >= d.other_bs ? std::vector<int64_t>(input_dims.begin(), input_dims.end() - 2)
                                                : std::vector<int64_t>(other_dims.begin(), other_dims.end() - 2);
    output_dims.push_back(d.m);
    output_dims.push_back(d.n);
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32, input->GetDevice());

    const auto *input_ptr = static_cast<const float *>(input->DataPtr());
    const auto *other_ptr = static_cast<const float *>(other->DataPtr());
    auto *output_ptr = static_cast<float *>(output->DataPtr());

    for (int64_t b = 0; b < d.bs; ++b) {
        ConstMatrixMap a(input_ptr + (d.input_bs == 1 ? 0 : b) * d.m * d.k, d.m, d.k);
        ConstMatrixMap w(other_ptr + (d.other_bs == 1 ? 0 : b) * d.k * d.n, d.k, d.n);
        MatrixMap o(output_ptr + b * d.m * d.n, d.m, d.n);
        o.noalias() = a * w;
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================

    /*
      grad_input[*, m, k] = grad_output[*, m, n] * other[*, k, n]^T
      grad_other[*, k, n] = input[*, m, k]^T * grad_output[*, m, n]
      若某一侧发生了 batch 广播，其梯度需要在 batch 维上累加，故先清零再用 += 累积。
    */
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto d = ComputeMatmulDims(input_dims, other_dims);
    CHECK_EQ(grad_output->NumElements(), d.bs * d.m * d.n);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32, other->GetDevice());
    grad_input->Fill<float>(0.0f);
    grad_other->Fill<float>(0.0f);

    const auto *input_ptr = static_cast<const float *>(input->DataPtr());
    const auto *other_ptr = static_cast<const float *>(other->DataPtr());
    const auto *grad_output_ptr = static_cast<const float *>(grad_output->DataPtr());
    auto *grad_input_ptr = static_cast<float *>(grad_input->DataPtr());
    auto *grad_other_ptr = static_cast<float *>(grad_other->DataPtr());

    for (int64_t b = 0; b < d.bs; ++b) {
        const int64_t input_b = d.input_bs == 1 ? 0 : b;
        const int64_t other_b = d.other_bs == 1 ? 0 : b;

        ConstMatrixMap a(input_ptr + input_b * d.m * d.k, d.m, d.k);
        ConstMatrixMap w(other_ptr + other_b * d.k * d.n, d.k, d.n);
        ConstMatrixMap go(grad_output_ptr + b * d.m * d.n, d.m, d.n);
        MatrixMap ga(grad_input_ptr + input_b * d.m * d.k, d.m, d.k);
        MatrixMap gw(grad_other_ptr + other_b * d.k * d.n, d.k, d.n);

        ga.noalias() += go * w.transpose();
        gw.noalias() += a.transpose() * go;
    }

    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    }

    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    if (bias) {
        grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL
