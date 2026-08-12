# TinyInfiniTrain 作业报告

## 一、test 通过截图

全部 8 个测例通过：

```
=============== ENVIRONMENT ===============
name, compute_cap, driver_version
NVIDIA A100-PCIE-40GB, 8.0, 580.142
nvcc  : Cuda compilation tools, release 13.0, V13.0.88
g++   : g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
cmake : cmake version 3.28.3

=============== make test ===============
Test project /root/TinyInfiniTrain/build/Release
    Start 1: test_elementwise
1/8 Test #1: test_elementwise .................   Passed    0.08 sec
    Start 2: test_matmul
2/8 Test #2: test_matmul ......................   Passed    0.07 sec
    Start 3: test_dispatcher
3/8 Test #3: test_dispatcher ..................   Passed    0.08 sec
    Start 4: test_tensor
4/8 Test #4: test_tensor ......................   Passed    0.07 sec
    Start 5: test_adam
5/8 Test #5: test_adam ........................   Passed    0.08 sec
    Start 6: test_gpt2
6/8 Test #6: test_gpt2 ........................   Passed   61.41 sec
    Start 7: test_matmul_cuda
7/8 Test #7: test_matmul_cuda .................   Passed    0.59 sec
    Start 8: test_adam_cuda
8/8 Test #8: test_adam_cuda ...................   Passed    0.46 sec

100% tests passed, 0 tests failed out of 8

Total Test time (real) =  62.84 sec
```

其中 `test_gpt2` 的关键输出：

```
The meaning of life is that it is continuous and could not have actually closed off to
anything but the woman.<|endoftext|>In Tokyo, a parliamentarian in what has become a
widely-respected and strong Japanese political think tank set up by Prime Minister
Shinzo Abe earlier this month offered a wall with a swastika printed on a wall
Logits validation passed with 100 samples
[       OK ] GPT2TrainingTest.LogitsConsistency (62940 ms)
[  PASSED  ] 1 test.
```

## 二、作业步骤

### 作业一：autograd机制调用Neg kernel的实现

难度：⭐

对应测例：`TEST(ElementwiseTest, NegForward)`，`TEST(ElementwiseTest, NegBackward)`

需要实现的代码块位置：`infini_train/src/autograd/elementwise.cc`

```c++
std::vector<std::shared_ptr<Tensor>> Neg::Forward(const std::vector<std::shared_ptr<Tensor>> &input_tensors) {
    CHECK_EQ(input_tensors.size(), 1);
    const auto &input = input_tensors[0];

    auto device = input->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegForward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(input)};
}

std::vector<std::shared_ptr<Tensor>> Neg::Backward(const std::vector<std::shared_ptr<Tensor>> &grad_outputs) {
    // d(-x)/dx = -1，反向不依赖前向的输入，因此无需 SetupContext 保存张量，
    // 设备信息直接取自 grad_output（测例会绕过 Forward 单独调用 Backward）。
    CHECK_EQ(grad_outputs.size(), 1);
    const auto &grad_output = grad_outputs[0];

    auto device = grad_output->GetDevice().Type();
    auto kernel = Dispatcher::Instance().GetKernel({device, "NegBackward"});
    return {kernel.Call<std::shared_ptr<Tensor>>(grad_output)};
}
```

#### 解决思路

这一题本质上是在理解框架的分层约定：**autograd 层不写任何数学，只负责"按设备取 kernel → 调用 kernel → 保存反向所需的张量"**。同文件里的 `Reciprocal`、`Sin` 已经给出了完整样板，照搬三行结构即可：

1. 从输入张量拿到 `DeviceType`；
2. 用 `{device, "NegForward"}` 作为 key 向 `Dispatcher` 单例要 kernel；
3. 通过 `KernelFunction::Call<返回类型>(参数...)` 调用。

真正的数学（逐元素取反）在 `kernels/cpu/elementwise.cc` 和 `kernels/cuda/elementwise.cu` 里已经给好了，CPU/CUDA 各一份，上层完全不感知设备差异。

#### 遇到问题

**问题一：`Forward` 里不能调用 `Apply`。**
一开始想当然地写成 `std::make_shared<Neg>()->Apply(...)`，但 `Function::Apply` 内部第一件事就是调用 `Forward`，这样会无限递归爆栈。`Forward` 是被 `Apply` 调用的下层函数，只能直接调 kernel。

**问题二：`Backward` 不能依赖 `saved_tensors_`。**
测例 `NegBackward` 是直接构造 `autograd::Neg neg_op;` 然后调用 `neg_op.Backward({grad_output})`，**完全没有走 `Forward`/`SetupContext`**。如果按照 `Reciprocal::Backward` 的写法去读 `saved_tensors_[0]`，会直接越界崩溃。

好在取反的导数是常数 −1，反向本来就不需要前向输入，`NegBackward` kernel 的签名也只接收 `grad_output`。所以这里的正确做法是：不实现 `SetupContext`，设备信息从 `grad_output` 取。这个细节也解释了为什么头文件里 `Neg` 类没有声明 `SetupContext` —— 框架作者是有意为之的。

### 作业二：实现矩阵乘法

难度：⭐⭐

#### CPU实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiply)`，`TEST(MatmulTest, BatchedMatrixMultiply)`, `TEST(MatmulTest, BackwardPass)`

需要实现的代码块位置：`infini_train/src/kernels/cpu/linear.cc`

```c++
namespace {
using RowMajorMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixMap = Eigen::Map<RowMajorMatrix>;
using ConstMatrixMap = Eigen::Map<const RowMajorMatrix>;

// 矩阵乘的形状信息：最后两维参与乘法，其余前置维度折叠成 batch。
struct MatmulDims {
    int64_t m = 0, k = 0, n = 0;
    int64_t input_bs = 1, other_bs = 1, bs = 1;
};

MatmulDims ComputeMatmulDims(const std::vector<int64_t> &input_dims, const std::vector<int64_t> &other_dims) {
    CHECK_GE(input_dims.size(), 2);
    CHECK_GE(other_dims.size(), 2);

    MatmulDims d;
    d.m = input_dims[input_dims.size() - 2];
    d.k = *input_dims.rbegin();
    CHECK_EQ(other_dims[other_dims.size() - 2], d.k);
    d.n = *other_dims.rbegin();

    d.input_bs = std::accumulate(input_dims.begin(), input_dims.end() - 2, int64_t{1}, std::multiplies<int64_t>{});
    d.other_bs = std::accumulate(other_dims.begin(), other_dims.end() - 2, int64_t{1}, std::multiplies<int64_t>{});
    d.bs = std::max(d.input_bs, d.other_bs);
    CHECK(d.input_bs == d.bs || d.input_bs == 1);
    CHECK(d.other_bs == d.bs || d.other_bs == 1);
    return d;
}
} // namespace

std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
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
    const auto &input_dims = input->Dims();
    const auto &other_dims = other->Dims();
    const auto d = ComputeMatmulDims(input_dims, other_dims);
    CHECK_EQ(grad_output->NumElements(), d.bs * d.m * d.n);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32, input->GetDevice());
    auto grad_other = std::make_shared<Tensor>(other_dims, DataType::kFLOAT32, other->GetDevice());
    grad_input->Fill<float>(0.0f);
    grad_other->Fill<float>(0.0f);

    /* ... 取指针 ... */
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
```

#### CUDA实现

对应测例：`TEST(MatmulTest, BasicMatrixMultiplyCuda)`,`TEST(MatmulTest, BatchedMatrixMultiplyCuda)`,`TEST(MatmulTest, BackwardPassCuda)`

需要实现的代码块位置：`infini_train/src/kernels/cuda/linear.cu`

```c++
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    /*
      input: (*, m, k) @ other: (*, k, n) -> output: (*, m, n)，数据均为行主序。
      cuBLAS 采用列主序，行主序矩阵 X(a, b) 在列主序下即 X^T(b, a)，因此
        output = input @ other   <==>   output^T = other^T @ input^T
      直接把 other 当作第一个操作数、input 当作第二个操作数传给 cublas 即可，无需显式转置。
    */
    const auto d = ComputeMatmulDims(input->Dims(), other->Dims());
    /* ... 组装 output_dims、分配 output ... */

    const float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    // 强制严格 FP32：禁用 TF32 张量核以及 split-k 等会改变累加顺序的快速路径
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
    /* ... 形状推导、分配 grad_input / grad_other ... */
    const float alpha = 1.0f;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

    // 广播的一侧需要在 batch 维累加，先清零再用 beta = 1 累积；未广播时直接覆盖写入
    const bool accumulate_grad_input = d.input_bs != d.bs;
    const bool accumulate_grad_other = d.other_bs != d.bs;
    if (accumulate_grad_input) { CUDA_CHECK(cudaMemset(grad_input_ptr, 0, grad_input->SizeInBytes())); }
    if (accumulate_grad_other) { CUDA_CHECK(cudaMemset(grad_other_ptr, 0, grad_other->SizeInBytes())); }
    const float beta_input = accumulate_grad_input ? 1.0f : 0.0f;
    const float beta_other = accumulate_grad_other ? 1.0f : 0.0f;

    for (int64_t b = 0; b < d.bs; ++b) {
        const int64_t input_b = d.input_bs == 1 ? 0 : b;
        const int64_t other_b = d.other_bs == 1 ? 0 : b;

        // grad_input = grad_output @ other^T
        // 列主序：C[k, m] = op(other)[k, n] * op(grad_output)[n, m]
        // other 行主序 (k, n) -> 列主序 (n, k)，需转置得到 (k, n)，故 opA = T，lda = n
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, d.k, d.m, d.n, &alpha,
                                 other_ptr + other_b * d.k * d.n, d.n, grad_output_ptr + b * d.m * d.n, d.n,
                                 &beta_input, grad_input_ptr + input_b * d.m * d.k, d.k));

        // grad_other = input^T @ grad_output
        // 列主序：C[n, k] = op(grad_output)[n, m] * op(input)[m, k]
        // input 行主序 (m, k) -> 列主序 (k, m)，需转置得到 (m, k)，故 opB = T，ldb = k
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, d.n, d.k, d.m, &alpha,
                                 grad_output_ptr + b * d.m * d.n, d.n, input_ptr + input_b * d.m * d.k, d.k,
                                 &beta_other, grad_other_ptr + other_b * d.k * d.n, d.n));
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    return {grad_input, grad_other};
}
```

#### 解决思路

矩阵乘的数学部分不难，`grad_input = grad_output · otherᵀ`、`grad_other = inputᵀ · grad_output`，难点全在**形状处理**和**行主序 / 列主序的转换**上。

**形状处理**：把"最后两维参与乘法、前置维度折叠成 batch"这个规则抽成一个 `ComputeMatmulDims` 辅助函数，前向和反向共用，避免两处推导不一致。同时支持一侧 batch 为 1 的广播（把该侧的 batch 偏移固定为 0），反向时对被广播的一侧在 batch 维累加梯度。

**行主序转列主序**：cuBLAS 是列主序，而框架里的张量是行主序。利用恒等式"行主序矩阵 X(a,b) 在列主序下就是 Xᵀ(b,a)"，`C = A·B` 等价于 `Cᵀ = Bᵀ·Aᵀ`，所以只要把 B、A 按顺序传进 `cublasSgemm`、并用 n/m/k 而不是 m/n/k 作为维度，就能得到正确的行主序结果，**一次转置都不用做**。这个技巧本文件里给定的 `LinearForward` 已经用了，可以直接参考。

#### 遇到问题

**问题一：不能直接使用 `Tensor::EigenMatrix()`。**
这是最坑的一点。`EigenMatrix()` 的实现是把"除最后一维外的所有维度折叠成行"：

```c++
const int64_t bs = std::accumulate(dims_.rbegin() + 1, dims_.rend(), 1, std::multiplies<int64_t>());
return Eigen::Map<...>(reinterpret_cast<float *>(DataPtr()), bs, *dims_.rbegin());
```

对于 `LinearForward` 那种 `(*, in_features) × (in_features, out_features)` 的场景这样折叠是对的，但测例 `BatchedMatrixMultiply` 传的是 `(2,2,3) × (2,3,2)`，用 `EigenMatrix()` 会被折叠成 `(4,3) × (6,2)`，形状直接对不上。正确做法是按 batch 自己构造 `Eigen::Map`，每个 batch 用 `ptr + b * m * k` 作为起始地址。

**问题二：反向传播中广播侧的梯度必须累加。**
如果某一侧的 batch 为 1 而另一侧为 N，前向时这一侧被复用了 N 次，反向时它的梯度就是 N 个 batch 的梯度之和。所以先 `Fill(0)` 再用 `+=`（CUDA 侧用 `beta = 1`）。GPT-2 的实际路径里两侧 batch 恒等，不走这条分支，但补上才是完整实现。

**问题三（CUDA）：数值精度与参考值对齐。**
这是整个作业里花时间最多的问题，详见作业六的记录。

### 作业三：实现Adam优化器

难度：⭐

#### CPU实现

对应测例：`TEST(AdamOptimizerTest, BasicParameterUpdate)`,`TEST(AdamOptimizerTest, MomentumAccumulation)`

代码位置：infini_train/src/kernels/cpu/accumulate_grad.cc

```c++
void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    const int64_t num_elements = param->NumElements();
    CHECK_EQ(grad->NumElements(), num_elements);
    CHECK_EQ(m->NumElements(), num_elements);
    CHECK_EQ(v->NumElements(), num_elements);

    const auto *grad_ptr = static_cast<const float *>(grad->DataPtr());
    auto *param_ptr = static_cast<float *>(param->DataPtr());
    auto *m_ptr = static_cast<float *>(m->DataPtr());
    auto *v_ptr = static_cast<float *>(v->DataPtr());

    const float bias_correction1 = 1.0f - std::pow(beta1, static_cast<float>(t));
    const float bias_correction2 = 1.0f - std::pow(beta2, static_cast<float>(t));

    for (int64_t idx = 0; idx < num_elements; ++idx) {
        const float g = grad_ptr[idx];
        m_ptr[idx] = beta1 * m_ptr[idx] + (1.0f - beta1) * g;
        v_ptr[idx] = beta2 * v_ptr[idx] + (1.0f - beta2) * g * g;

        const float m_hat = m_ptr[idx] / bias_correction1;
        const float v_hat = v_ptr[idx] / bias_correction2;
        param_ptr[idx] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }
}
```

#### CUDA实现

代码位置：infini_train/src/kernels/cuda/accumulate_grad.cu

```c++
__global__ void AdamAccumulateGradKernel(const float *grad_ptr, float *param_ptr, float *m_ptr, float *v_ptr,
                                         size_t num_elements, float learning_rate, float beta1, float beta2, float eps,
                                         float bias_correction1, float bias_correction2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        const float g = grad_ptr[idx];

        const float m_new = beta1 * m_ptr[idx] + (1.0f - beta1) * g;
        const float v_new = beta2 * v_ptr[idx] + (1.0f - beta2) * g * g;
        m_ptr[idx] = m_new;
        v_ptr[idx] = v_new;

        const float m_hat = m_new / bias_correction1;
        const float v_hat = v_new / bias_correction2;
        param_ptr[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + eps);
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    size_t num_elements = param->NumElements();
    /* ... CHECK 与取指针 ... */

    // 偏置校正项与元素无关，放在 host 侧算好再传入，避免每个线程重复求幂
    const float bias_correction1 = 1.0f - powf(beta1, static_cast<float>(t));
    const float bias_correction2 = 1.0f - powf(beta2, static_cast<float>(t));

    int threads_per_block = 256;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    AdamAccumulateGradKernel<<<num_blocks, threads_per_block>>>(grad_ptr, param_ptr, m_ptr, v_ptr, num_elements,
                                                               learning_rate, beta1, beta2, eps, bias_correction1,
                                                               bias_correction2);
}
```

#### 解决思路

标准 Adam 的四行公式，逐元素独立，是最适合 GPU 的算子形态：

```
m_t   = β₁·m_{t-1} + (1-β₁)·g
v_t   = β₂·v_{t-1} + (1-β₂)·g²
m̂ = m_t/(1-β₁ᵗ),  v̂ = v_t/(1-β₂ᵗ)
θ    -= lr · m̂ / (√v̂ + ε)
```

两个工程上的小考虑：

1. **偏置校正项提到循环/kernel 外面算**。`1-β₁ᵗ` 只和步数 `t` 有关，与元素无关，放在 host 侧算一次，避免每个线程都调一次 `powf`。
2. **CUDA kernel 里用局部变量中转**。先算出 `m_new`/`v_new` 再写回全局内存，然后直接用局部变量算 `m_hat`，比"写回后再读回来"少两次全局内存访问。

#### 遇到问题

**问题一：`ε` 的位置必须在 `sqrt` 外面。**
Adam 有两种常见写法：`m̂/(√v̂ + ε)` 和 `m̂/√(v̂ + ε)`，PyTorch 用的是前者。测例 `MomentumAccumulation` 用 `EXPECT_NEAR(..., 1e-5)` 逐步核对了三步更新，写成后者会直接对不上。测例里的参考实现写得很明确：

```c++
expected_update -= learning_rate * m_hat / (std::sqrt(v_hat) + 1e-8f);
```

**问题二：`t` 从 1 开始而不是 0。**
`Adam::Step()` 里是 `++t_;` 在前、调用 kernel 在后，所以传进 kernel 的第一步 `t` 就是 1。如果误以为从 0 开始，`1-β⁰ = 0` 会导致除零，第一步直接出 `inf`/`nan`。

### 作业四：实现Tensor基础操作

#### 实现Tensor的Flatten操作

难度：⭐

对应测例：`TEST(TensorTransformTest, Flatten2DTo1D)`,`TEST(TensorTransformTest, FlattenWithRange)`,`TEST(TensorTransformTest, FlattenNonContiguous)`

代码位置：infini_train/src/tensor.cc

```c++
std::shared_ptr<Tensor> Tensor::Flatten(int64_t start, int64_t end) {
    const int64_t rank = static_cast<int64_t>(dims_.size());
    // 标量按 PyTorch 语义展平成一维
    if (rank == 0) {
        return Contiguous()->View({1});
    }

    // 支持负数索引
    if (start < 0) { start += rank; }
    if (end < 0) { end += rank; }
    CHECK_GE(start, 0);
    CHECK_LT(start, rank);
    CHECK_GE(end, 0);
    CHECK_LT(end, rank);
    CHECK_LE(start, end) << "flatten range [" << start << ", " << end << "] is invalid";

    std::vector<int64_t> new_shape;
    new_shape.reserve(rank - (end - start));
    new_shape.insert(new_shape.end(), dims_.begin(), dims_.begin() + start);
    new_shape.push_back(
        std::accumulate(dims_.begin() + start, dims_.begin() + end + 1, int64_t{1}, std::multiplies<int64_t>{}));
    new_shape.insert(new_shape.end(), dims_.begin() + end + 1, dims_.end());

    // 先 Contiguous 保证底层数据按行优先连续排布，再改写形状
    return Contiguous()->View(new_shape);
}
```

#### 实现Tensor的反向传播机制

难度：⭐

对应测例：`TEST(TensorAutogradTest, BackwardComputesGradient)`,`TEST(TensorAutogradTest, BackwardWithMultipleOutputs)`

代码位置：infini_train/src/tensor.cc

```c++
void Tensor::Backward(std::shared_ptr<Tensor> gradient, bool retain_graph, bool create_graph) const {
    // 叶子张量没有 grad_fn，反向图到此为止
    if (!grad_fn_) {
        return;
    }

    // 未显式指定初始梯度时默认为全 1（等价于对 sum 求导）
    if (!gradient) {
        gradient = std::make_shared<Tensor>(dims_, dtype_, GetDevice());
        gradient->Fill<float>(1.0f);
    }
    CHECK_EQ(gradient->NumElements(), num_elements_) << "gradient shape mismatch in Backward";

    // 交给反向引擎：Function::BackwardPartial 内部按依赖计数触发本节点的 Backward，
    // 并沿 next_functions_ 递归回传，叶子节点的 AccumulateGrad 会把梯度累加进 grad_。
    // 多次调用 Backward（多输出场景）会累加到同一份叶子梯度上。
    grad_fn_->BackwardPartial(gradient, output_idx_);
}
```

#### 解决思路

**Flatten** 的实现其实注释里已经给出了答案（`return Contiguous()->View(new_shape);`），关键是把 `new_shape` 算对：`[0, start)` 原样保留 → `[start, end]` 连乘成一维 → `(end, rank)` 原样保留，再处理负索引和标量两个边界情况。

**Backward** 这一题看起来只有三行，但前提是先读懂框架的反向引擎设计。这个框架用的**不是**常见的"拓扑排序 + 队列"，而是一套**引用计数驱动的推送式（push-based）反向传播**：

- **前向建图**（`Function::Apply`）：遍历输入张量，若是需要梯度的叶子节点，就挂一个 `AccumulateGrad` 节点直接写进 `tensor->grad()`；若是中间结果，则记录其 `grad_fn` 并调用 `IncreaseDependenciesNumber()` 给上游的引用计数加一。
- **反向触发**（`Function::BackwardPartial`）：每个节点收到一份梯度就累计一次，只有当**本节点所有输出的梯度都到齐**且**所有下游消费者都已回传**（`dependencies_reached_ == dependencies_number_`）时，才真正执行 `Backward()`，然后沿 `next_functions_` 推给上游。

理解了这一点，`Tensor::Backward` 要做的就只有三件事：叶子直接返回、缺省梯度补全 1、把梯度交给 `grad_fn_->BackwardPartial(gradient, output_idx_)`。**多输出的梯度累加是引擎负责的，不需要自己写累加逻辑** —— 这也是测例 `BackwardWithMultipleOutputs` 想考的点。

#### 遇到问题

**问题一：`Backward` 是 `const` 成员函数。**
签名里带 `const`，一开始以为不能修改任何状态。但实际上 `grad_fn_` 是 `std::shared_ptr`，`const shared_ptr` 的 `operator->` 返回的仍是非 const 的 `Function*`，通过它调用非 const 成员是合法的；叶子的梯度也是通过 `grad_` 指向的对象修改的，不涉及 `Tensor` 自身成员的写入。

**问题二：多输出累加的验证。**
测例 `BackwardWithMultipleOutputs` 里 `y1 = x*2`、`y2 = x³`，分别以梯度 1 和 2 反传，期望 `x.grad = 1*2 + 2*3 = 8`。一开始担心需要在 `Tensor::Backward` 里手动累加，实际上不需要：两次 `Apply` 各自创建了一个 `AccumulateGrad` 节点，但它们**包裹的是同一个 `x->grad()` 张量**，累加天然就发生在那里。

**问题三：`FlattenNonContiguous` 测例的命名有点误导。**
测例先做 `Transpose(0,1)` 再 `Flatten`。但这个框架的 `Tensor` **没有 stride 字段**，`Transpose` 不是零拷贝视图，而是由 `transform.cc` 里真正搬运数据的 kernel 产出的一个全新连续张量。所以 `Contiguous()->View()` 这条路径本来就是成立的，不需要额外处理非连续内存。

### 作业五 注册算子kernel的实现

难度：⭐⭐⭐

对应测例：`TEST(DispatcherTest, RegisterAndGetKernel)`,`TEST(DispatcherTest, DuplicateRegistration)`,`TEST(DispatcherTest, GetNonexistentKernel)`

代码位置：infini_train/include/dispatcher.h

```c++
template <typename RetT, class... ArgsT> RetT Call(ArgsT... args) const {
    using FuncT = RetT (*)(ArgsT...);
    CHECK(func_ptr_ != nullptr) << "Calling an empty kernel function";
    return reinterpret_cast<FuncT>(func_ptr_)(std::forward<ArgsT>(args)...);
}

template <typename FuncT> void Register(const KeyT &key, FuncT &&kernel) {
    CHECK(!key_to_kernel_map_.count(key))
        << "Kernel already registered: " << key.second << " on device: " << static_cast<int>(key.first);
    key_to_kernel_map_.emplace(key, KernelFunction(std::forward<FuncT>(kernel)));
}

// 两层间接展开，保证 __COUNTER__ 先求值再拼接，避免同一作用域内多次注册时变量重名。
#define INFINI_TRAIN_CONCAT_IMPL(lhs, rhs) lhs##rhs
#define INFINI_TRAIN_CONCAT(lhs, rhs) INFINI_TRAIN_CONCAT_IMPL(lhs, rhs)

// 借助静态存储期变量的初始化，在 main 之前完成注册。
// 展开为一条声明语句，因此在命名空间作用域与函数作用域下均可使用。
#define REGISTER_KERNEL(device, kernel_name, kernel_func)                                                              \
    static const bool INFINI_TRAIN_CONCAT(kInfiniTrainKernelRegistered_, __COUNTER__) = []() {                         \
        ::infini_train::Dispatcher::Instance().Register({device, #kernel_name}, kernel_func);                          \
        return true;                                                                                                   \
    }();
```

#### 解决思路

`Dispatcher` 是整个框架的枢纽，也是本次作业里设计感最强的一部分。它用**手写的类型擦除**实现了零开销的多设备分发：把任意签名的 kernel 函数指针 `reinterpret_cast` 成 `void*` 存起来，调用时再用模板参数把签名还原回来。没有虚函数、没有 `std::function`、没有 RTTI。

三个部分分别对应：

1. **`Call`**：脚手架已经给出 `using FuncT = RetT (*)(ArgsT...);`，剩下就是把 `void*` 转回 `FuncT` 并调用。注意 `RetT = void` 时 `return f(...)` 也是合法的，不需要特判。
2. **`Register`**：先查重再插入。重复注册必须 `CHECK` 失败，且消息里要包含 `"Kernel already registered"` —— 测例用 `EXPECT_DEATH` 正则匹配这个字符串。
3. **`REGISTER_KERNEL` 宏**：定义一个静态存储期的 `bool` 变量，用一个 lambda 的返回值初始化它，从而在 `main` 之前完成注册。

还有一个容易被忽略但至关重要的配合点：kernel 注册依赖静态对象的初始化，而静态库里没有被显式引用的目标文件会被链接器丢弃。`CMakeLists.txt` 里的 `-Wl,--whole-archive` 就是为此存在的 —— 没有这一行，所有 `REGISTER_KERNEL` 都不会生效，运行时会全部报 "Kernel not found"。

#### 遇到问题

**问题一：宏必须在函数作用域内也能用。**
`test_dispatcher.cc` 是在 `TEST(...)` 的函数体内部调用 `REGISTER_KERNEL` 的，而 kernel 源文件里是在命名空间作用域调用的。所以宏体必须是一条在两种作用域下都合法的声明语句 —— `static const bool x = ...;` 满足这个条件。

**问题二：同一作用域内多次展开会变量重名。**
`TEST(DispatcherTest, DuplicateRegistration)` 在同一个函数里展开了两次宏。如果变量名写死，会直接编译报重定义。用 `__COUNTER__` 拼接可以解决，但必须用**两层宏间接展开**，否则 `__COUNTER__` 会被当作字面量拼接成 `kXxx__COUNTER__`。

**问题三：原始宏是"空宏"，这是个很好的调试起点。**
脚手架里的宏定义是这样的：

```c++
#define REGISTER_KERNEL(device, kernel_name, kernel_func) \
    // =================================== 作业 ===================================
```

由于反斜杠续行发生在注释剥离**之前**，这一整行会被拼接后整体变成注释，宏体为空。所以在完成这一题之前，整个框架**一个 kernel 都没有注册**，任何 `GetKernel` 都会 `CHECK` 失败 —— 这解释了为什么作业五是其它所有题目的硬前置。

**问题四：`Call` 的类型安全完全靠约定。**
调用点写 `Call<std::shared_ptr<Tensor>>(input)`，模板会推导出 `ArgsT = std::shared_ptr<Tensor>`（按值），而真实 kernel 的签名是 `const std::shared_ptr<Tensor>&`。二者签名并不一致，之所以能正常工作，是因为在 Itanium ABI 下带非平凡析构的类类型参数一律按"隐式引用"传递，按值和 const 引用的 ABI 表示相同。这是这套设计里最脆弱的地方：**签名写错不会有任何编译错误，只会在运行时炸掉**。

### 作业六：实现GPT-2整体训练

难度：⭐⭐⭐⭐

对应测例：`TEST_F(GPT2TrainingTest, LogitsConsistency)`

#### 训练过程logits对比

在 A100-PCIE-40GB（sm_80）上：

```
Logits validation passed with 100 samples
[       OK ] GPT2TrainingTest.LogitsConsistency (62940 ms)
```

#### 数据读取实现

代码位置：example/common/tiny_shakespeare_dataset.cc

```c++
TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    constexpr size_t kHeaderSizeInBytes = 1024;

    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open dataset file: " << path;

    const auto header = ReadSeveralBytesFromIfstream(kHeaderSizeInBytes, &ifs);
    const int32_t magic = BytesToType<int32_t>(header, 0);
    const int32_t version = BytesToType<int32_t>(header, 4);
    const int32_t num_tokens = BytesToType<int32_t>(header, 8);
    CHECK(kTypeMap.count(magic)) << "Unsupported dataset magic number: " << magic;
    CHECK_GT(num_tokens, 0) << "Empty dataset file: " << path;

    const auto type = kTypeMap.at(magic);
    const size_t token_size_in_bytes = kTypeToSize.at(type);

    // 按 sequence_length 切成整行，多余的尾部 token 丢弃。
    // 注意最后一行只用于给倒数第二行提供右移一位的标签，因此可用样本数为 num_sequences - 1。
    const int64_t num_sequences = static_cast<int64_t>(num_tokens / sequence_length);
    CHECK_GE(num_sequences, 2) << "dataset too small for sequence_length=" << sequence_length;

    TinyShakespeareFile result;
    result.type = type;
    result.dims = {num_sequences, static_cast<int64_t>(sequence_length)};

    // 下游 EmbeddingForward 要求索引张量为 INT64，这里统一把 uint16/uint32 提升为 int64 存放
    result.tensor = infini_train::Tensor(result.dims, DataType::kINT64);
    const size_t num_elements = static_cast<size_t>(num_sequences) * sequence_length;
    const auto raw = ReadSeveralBytesFromIfstream(num_elements * token_size_in_bytes, &ifs);
    auto *dst = static_cast<int64_t *>(result.tensor.DataPtr());
    for (size_t idx = 0; idx < num_elements; ++idx) {
        dst[idx] = type == TinyShakespeareType::kUINT16
                     ? static_cast<int64_t>(BytesToType<uint16_t>(raw, idx * token_size_in_bytes))
                     : static_cast<int64_t>(BytesToType<uint32_t>(raw, idx * token_size_in_bytes));
    }
    return result;
}

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    // 三个成员均为 const，只能在初始化列表中赋值；初始化顺序与声明顺序一致，
    // text_file_ 先于 num_samples_ 构造，因此可以直接用它的 dims 推导样本数。
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)), sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),
      num_samples_(static_cast<size_t>(text_file_.dims[0]) - 1) {
    CHECK_EQ(text_file_.dims.size(), 2);
    CHECK_EQ(static_cast<size_t>(text_file_.dims[1]), sequence_length_);
}
```

#### Tokenizer功能实现

代码位置：example/common/tokenizer.cc

```c++
Tokenizer::Tokenizer(const std::string &filepath) {
    constexpr size_t kHeaderSizeInBytes = 1024;

    std::ifstream ifs(filepath, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open tokenizer file: " << filepath;

    const auto header = ReadSeveralBytesFromIfstream(kHeaderSizeInBytes, &ifs);
    magic_number_ = BytesToType<uint32_t>(header, 0);
    CHECK(kEotMap.count(magic_number_)) << "Unsupported tokenizer magic number: " << magic_number_;
    const auto version = static_cast<Version>(BytesToType<uint32_t>(header, 4));
    vocab_size_ = BytesToType<uint32_t>(header, 8);

    // v1 的头部没有 eot 字段，按 magic 查表；v2 直接从头部第 4 个 int32 读取
    eot_token_ = version == Version::kV2 ? BytesToType<uint32_t>(header, 12) : kEotMap.at(magic_number_);

    // 词表区：每个 token 由 1 字节长度 + 该长度的原始字节组成
    token_table_.reserve(vocab_size_);
    for (uint32_t idx = 0; idx < vocab_size_; ++idx) {
        const auto length_bytes = ReadSeveralBytesFromIfstream(1, &ifs);
        const size_t length = static_cast<size_t>(length_bytes[0]);
        const auto token_bytes = ReadSeveralBytesFromIfstream(length, &ifs);
        token_table_.emplace_back(reinterpret_cast<const char *>(token_bytes.data()), length);
    }
    CHECK_EQ(token_table_.size(), vocab_size_);
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    if (token_id >= vocab_size_) {
        LOG(ERROR) << "invalid token id: " << token_id << " (vocab_size=" << vocab_size_ << ")";
        return "";
    }
    return token_table_[token_id];
}
```

```c++
void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    /* ...原代码... */
    // 原写法 `uint64_t kRngState = kRngState;` 是自我初始化（未定义行为），
    // 且会遮蔽文件顶部的常量种子，这里改名并用常量种子显式初始化。
    uint64_t rng_state = kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        // (bs, seq_len) -> GPT2 -> (bs, seq_len, vocab_size)
        auto logits = model.Forward({x})[0];
        // 在词表维做 softmax 得到概率分布
        auto probs = nn::function::Softmax(logits, -1);
        // 采样只在 CPU 上做，把结果搬回主机
        auto probs_cpu = probs->To(Device());

        const int64_t vocab_size = *probs_cpu.Dims().rbegin();
        // 取 batch 0 中位置 t-1 的分布，用它预测第 t 个 token
        auto *probs_ptr = static_cast<float *>(probs_cpu.DataPtr()) + static_cast<int64_t>(t - 1) * vocab_size;

        const float coin = RandomF32(rng_state);
        const uint32_t next_token = static_cast<uint32_t>(SampleMult(probs_ptr, vocab_size, coin));

        std::cout << Decode(next_token) << std::flush;

        // 把采样结果写回输入序列（各 batch 保持一致），再同步到设备上继续下一步
        for (int b = 0; b < batch_size; ++b) { x_buff[b * sequence_length + t] = next_token; }
        x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    }
    std::cout << std::endl;
}
```

#### 解决思路

这一题最有意思的地方在于，**题面没有说清楚的信息，全都能从已有代码里反推出来**。

**数据集的张量 dtype 必须是 INT64。** 文件里的 GPT-2 token 是 `uint16`，而文件顶部的 `kTypeToDataType` 映射写的是 `kUINT16 → DataType::kUINT16`，很容易照着填。但有两条硬证据说明必须存成 `int64`：

1. `kernels/cpu/embedding.cc` 里 `EmbeddingForward` 第一行就是 `CHECK(input->Dtype() == DataType::kINT64);`
2. 已给出的 `operator[]` 里，标签 `y` 的偏移是 `idx * sequence_size_in_bytes_ + sizeof(int64_t)` —— "错开一个元素"意味着元素宽度就是 8 字节

**样本数是 `dims[0] - 1` 而不是 `dims[0]`。** 同样从 `operator[]` 的 `CHECK_LT(idx, text_file_.dims[0] - 1)` 反推：最后一行数据不作为独立样本，只用来给倒数第二行提供右移一位的标签 `y`。

**`sequence_size_in_bytes_ = sequence_length * sizeof(int64_t)`**，由上面两点直接推出。

**Tokenizer 的文件格式**参考 llm.c 的 `tokenizer_init`：1024 字节头（magic / version / vocab_size / 可选 eot），随后每个 token 是"1 字节长度 + 变长字节内容"。`Version::kV1` 的头部没有 eot 字段，需要按 magic 查文件里已经给好的 `kEotMap`。

**文本生成**是标准的自回归采样循环：前向拿 logits → 在词表维 softmax → 取位置 `t-1` 的分布 → 用 `RandomF32` + `SampleMult` 做多项式采样 → `Decode` 输出 → 把新 token 写回输入序列，继续下一步。

#### 遇到问题

**问题一：三个成员是 `const`，只能在初始化列表里赋值。**

```c++
const size_t sequence_length_ = 0;
const size_t sequence_size_in_bytes_ = 0;
const size_t num_samples_ = 0;
```

不能在构造函数体里赋值，必须全部走成员初始化列表。同时要注意初始化顺序**按声明顺序**而非初始化列表的书写顺序，好在 `text_file_` 声明在最前面，所以 `num_samples_` 可以安全地用 `text_file_.dims[0]`。

**问题二：`GenerateText` 里给定代码有一处未定义行为。**

```c++
uint64_t kRngState = kRngState;   // 用自己初始化自己
```

这行同时做了两件坏事：用未初始化的自身值初始化自己（UB），以及遮蔽了文件顶部的 `constexpr uint64_t kRngState = 1337;`。实测会导致采样种子随机、生成结果不可复现。改名为 `rng_state` 并显式用常量种子初始化后正常。

**问题三（最耗时）：logits 数值与参考值对不齐。**

代码全部写完后，前 7 个测例一次通过，但 `test_gpt2` 在 RTX 3080 上始终差一点：

```
Logits mismatch at position 385973: Reference=-204.265, Current=-204.264, Diff=0.00149536
```

容差是 `1e-3`，而 100 个采样点里只有 1 个超标，超出幅度约 50%。相对误差只有 `1.5e-3 / 204 ≈ 7.3e-6`，属于 float32 的累加误差量级，说明**逻辑是对的，差异来自浮点累加顺序**。逐步排查：

| 措施 | Diff |
|---|---|
| 初版（`cublasSgemmStridedBatched`，默认 math mode） | 0.001495 |
| `NVIDIA_TF32_OVERRIDE=0` 全局关闭 TF32 | 0.001328 |
| 改为逐 batch `cublasSgemm` + `CUBLAS_PEDANTIC_MATH` | 0.001205 |
| 上述两者叠加 | 0.001144 |

每一层都在收敛，但始终跨不过 `1e-3`。定位到根因是**GPU 架构**：`CMakeLists.txt` 里写的是 `CUDA_ARCHITECTURES "75;80"`，也就是项目只针对 sm_75（Turing）和 sm_80（A100）验证过，参考 logits 大概率就是在这两种卡上生成的；而 RTX 3080 是 sm_86，cuBLAS 在其上选择的 GEMM 内核不同，累加顺序随之改变。

把同一份代码、同一个 commit 换到 **A100-PCIE-40GB（sm_80）** 上，一次通过：

```
Logits validation passed with 100 samples
[       OK ] GPT2TrainingTest.LogitsConsistency (62940 ms)
```

这个过程也印证了一点：这类"与参考值逐点比对"的端到端测例，对硬件和数学库的选核策略是敏感的，跨架构复现时需要把 GPU 架构本身当作变量来考虑。

## 三、构建与环境说明

项目对工具链的要求（见 `docs/项目部署.md` 与 `CMakeLists.txt`）比较严格，实际踩到的几个点：

| 依赖 | 要求 | 说明 |
|---|---|---|
| CMake | **≥ 3.28** | `CMakeLists.txt` 里写的是 `cmake_minimum_required(VERSION 3.28)`，比部署文档写的 3.13 高 |
| GCC/G++ | **≥ 13** | `example/gpt2/net.cc` 用了 `std::format`，而 `example_gpt2` 库是所有测试目标都要链接的，绕不开 |
| CUDA | **≥ 12** | `dispatcher.h` 用到 C++20 的 `std::map::contains`，nvcc 11.x 最高只支持 C++17，会直接编译失败 |
| submodule | 必须递归拉取 | `.gitmodules` 用的是 `git@github.com:` SSH 地址，无 GitHub 密钥时需配 `git config --global url."https://github.com/".insteadOf "git@github.com:"` |

最省事的组合是 **Ubuntu 24.04 + CUDA 12.x/13.x devel 镜像**：gcc 13.3 和 cmake 3.28 都在默认源里，开箱即用。Ubuntu 22.04 则需要额外装 `ppa:ubuntu-toolchain-r/test` 的 gcc-13 并手动下载新版 CMake。

本次最终验证环境：

```
GPU    : NVIDIA A100-PCIE-40GB (compute capability 8.0)
CUDA   : 13.0
g++    : 13.3.0
CMake  : 3.28.3
OS     : Ubuntu 24.04.4 LTS
```

另外记录一个与作业无关但会影响自测的细节：`test/example/test_gpt2.cc` 里 `device_flag` 被硬编码为 `"cuda"`，因此 `make USE_CUDA=OFF` 的纯 CPU 构建**无法通过 `test_gpt2`**（`TensorBuffer` 会在 `kCUDA` 分支上 `LOG(FATAL)`）。而 `.github/workflows/build.yml` 恰好是 `make USE_CUDA=OFF` 之后跑 `make test-cpp`，两者是矛盾的。纯 CPU 环境下自测时，建议用 `ctest -E test_gpt2` 排除该用例，其余 5 个测例可以正常验证。
