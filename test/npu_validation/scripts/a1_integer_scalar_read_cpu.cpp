/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Appended to the unmodified generated pointer-ABI function. No tolerance.
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
using ReadType = READ_TYPE;
constexpr unsigned columns = 128;
constexpr unsigned guard = 16;
constexpr uint32_t sentinel = 0xa55ac33cU;
constexpr std::array<uint32_t, 16> patterns = {
    0U, 1U, 0x7fffffffU, 0x80000000U, 0x80000001U, 0xffffffffU, 0x01020304U, 0xfedcba98U,
    0x7fc01234U, 0xffc05678U, 0x7f800001U, 0x7f800000U, 0xff800000U, 0x00000001U, 0x00800000U, 0x80000001U};

void FillInput(std::vector<ReadType>& input, unsigned dataset) {
    for (unsigned i = 0; i < input.size(); ++i) {
        // Unsigned modular noise intentionally covers raw integer payload bits.
        const uint32_t bits = dataset < 2 ? patterns[(i + dataset) % patterns.size()] :
            patterns[(i + dataset) % patterns.size()] ^ (i * 1664525U + dataset);
        input[i] = std::bit_cast<ReadType>(bits);
    }
}

bool CheckOutput(const std::vector<ReadType>& input, const std::vector<float>& output, unsigned dataset) {
    for (unsigned i = 0; i < output.size(); ++i) {
        const bool active = i >= guard && i < guard + columns;
        const uint32_t expected = active ?
            std::bit_cast<uint32_t>(input[guard + 2 * ((i - guard) % 64) + 1]) : sentinel;
        const bool wordMatches = std::bit_cast<uint32_t>(output[i]) == expected;
        if (!wordMatches) {
            std::fprintf(stderr, "integer scalar raw word/guard mismatch at %u, dataset %u\n", i, dataset);
            return false;
        }
    }
    return true;
}

int CheckDataset(unsigned dataset) {
    std::vector<ReadType> input(columns + 2 * guard);
    std::vector<float> output(columns + 2 * guard, std::bit_cast<float>(sentinel));
    FillInput(input, dataset);
    const auto original = input;
    a1_integer_scalar_read(output.data() + guard, input.data() + guard);
    if (input != original) {
        return 1;
    }
    return CheckOutput(input, output, dataset) ? 0 : 2;
}
} // namespace

int main() {
    for (unsigned dataset = 0; dataset < 6; ++dataset) {
        const int result = CheckDataset(dataset);
        if (result != 0) {
            return result;
        }
    }
    return 0;
}
