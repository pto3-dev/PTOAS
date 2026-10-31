/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Appended to unmodified generated C++; the build selects FP16 or FP32 storage.
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
using Data = ROW_VIEW_CTYPE;
using Bits = ROW_VIEW_BITS;
constexpr unsigned count = 256, piece = 64, guard = 16;
constexpr Bits sentinel = static_cast<Bits>(0xa55ac33cU);

Bits pattern(unsigned index, unsigned dataset)
{
    constexpr std::array<uint32_t, 8> fp32 = {
        0U, 0x80000000U, 0x7fc01234U, 0xffc05678U, 0x7f800001U, 0xffffffffU, 0x7fffffffU, 0x01020304U};
    constexpr std::array<uint16_t, 8> fp16 = {
        0U, 0x8000U, 0x7e12U, 0xfe56U, 0x7c01U, 0xffffU, 0x7bffU, 0x0102U};
    const uint32_t value = sizeof(Bits) == 4 ? fp32[(index + dataset) % fp32.size()] :
                                              fp16[(index + dataset) % fp16.size()];
    return static_cast<Bits>(dataset < 2 ? value : value ^ index * 1664525U);
}

std::vector<Bits> rawBits(const std::vector<Data>& values)
{
    std::vector<Bits> result;
    result.reserve(values.size());
    for (Data value : values) {
        result.push_back(std::bit_cast<Bits>(value));
    }
    return result;
}

bool runDataset(unsigned dataset)
{
    std::vector<Data> input(count + 2 * guard), replacement(piece + 2 * guard);
    std::vector<Data> output(count + piece + 2 * guard, std::bit_cast<Data>(sentinel));
    for (unsigned i = 0; i < input.size(); ++i) {
        input[i] = std::bit_cast<Data>(pattern(i, dataset));
    }
    for (unsigned i = 0; i < replacement.size(); ++i) {
        replacement[i] = std::bit_cast<Data>(pattern(i * 7, dataset));
    }
    const auto original = rawBits(input), savedReplacement = rawBits(replacement);
    a1_reshape_rows(output.data() + guard, input.data() + guard, replacement.data() + guard);
    const bool changedInput = rawBits(input) != original || rawBits(replacement) != savedReplacement;
    if (changedInput) {
        return false;
    }
    std::vector<Bits> expected(output.size(), sentinel);
    for (unsigned i = 0; i < piece; ++i) {
        expected[guard + i] = original[guard + piece + i];
    }
    for (unsigned i = 0; i < count; ++i) {
        expected[guard + piece + i] = i >= piece && i < 2 * piece ?
            savedReplacement[guard + i - piece] : original[guard + i];
    }
    return rawBits(output) == expected;
}
} // namespace

int main()
{
    for (unsigned dataset = 0; dataset < 6; ++dataset) {
        if (!runDataset(dataset)) {
            std::fprintf(stderr, "reshape row alias dataset %u mismatch or guard corruption\n", dataset);
            return 1;
        }
    }
    return 0;
}
