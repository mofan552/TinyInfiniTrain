#include <cmath>
#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================

    /*
      m_t     = beta1 * m_{t-1} + (1 - beta1) * g_t
      v_t     = beta2 * v_{t-1} + (1 - beta2) * g_t^2
      m_hat   = m_t / (1 - beta1^t)
      v_hat   = v_t / (1 - beta2^t)
      param  -= learning_rate * m_hat / (sqrt(v_hat) + eps)
      注意 t 从 1 开始计数（Adam::Step 中先自增再调用），eps 加在 sqrt 之外。
    */
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

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
