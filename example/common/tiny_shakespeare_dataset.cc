#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
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

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */

    constexpr size_t kHeaderSizeInBytes = 1024;

    std::ifstream ifs(path, std::ios::binary);
    CHECK(ifs.is_open()) << "Failed to open dataset file: " << path;

    const auto header = ReadSeveralBytesFromIfstream(kHeaderSizeInBytes, &ifs);
    const int32_t magic = BytesToType<int32_t>(header, 0);
    const int32_t version = BytesToType<int32_t>(header, 4);
    const int32_t num_tokens = BytesToType<int32_t>(header, 8);
    CHECK(kTypeMap.count(magic)) << "Unsupported dataset magic number: " << magic;
    CHECK_GT(num_tokens, 0) << "Empty dataset file: " << path;
    LOG(INFO) << "dataset " << path << ": magic=" << magic << " version=" << version << " num_tokens=" << num_tokens;

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
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    // 三个成员均为 const，只能在初始化列表中赋值；初始化顺序与声明顺序一致，
    // text_file_ 先于 num_samples_ 构造，因此可以直接用它的 dims 推导样本数。
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)), sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),
      num_samples_(static_cast<size_t>(text_file_.dims[0]) - 1) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================

    CHECK_EQ(text_file_.dims.size(), 2);
    CHECK_EQ(static_cast<size_t>(text_file_.dims[1]), sequence_length_);
    LOG(INFO) << "TinyShakespeareDataset: " << num_samples_ << " samples, sequence_length=" << sequence_length_;
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
