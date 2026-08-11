#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */

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
    LOG(INFO) << "Tokenizer loaded: magic=" << magic_number_ << " vocab_size=" << vocab_size_
              << " eot_token=" << eot_token_;
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */

    if (token_id >= vocab_size_) {
        LOG(ERROR) << "invalid token id: " << token_id << " (vocab_size=" << vocab_size_ << ")";
        return "";
    }
    return token_table_[token_id];
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    // 原写法 `uint64_t kRngState = kRngState;` 是自我初始化（未定义行为），
    // 且会遮蔽文件顶部的常量种子，这里改名并用常量种子显式初始化。
    uint64_t rng_state = kRngState;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */

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
} // namespace infini_train
