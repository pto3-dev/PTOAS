/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Independent raw-word row-selection oracle; no NPU calls or tolerances.
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
constexpr unsigned sourceRows = BOUNDED_ROWS;
constexpr unsigned columns = BOUNDED_COLS;
constexpr unsigned choices = 8;
constexpr unsigned guard = 16;
constexpr uint32_t sentinel = 0xa55ac33cU;
constexpr std::array<uint32_t, 16> patterns = {
    0U, 1U, 255U, 256U, 335U, 336U, 8399U, 8400U,
    0x7fffffffU, 0x80000000U, 0x80000001U, 0xffffffffU,
    0x0000ffffU, 0x00010000U, 0x7fc01234U, 0xffc05678U};

bool sameWords(const std::vector<float>& a, const std::vector<float>& b)
{
    const bool sameSize = a.size() == b.size();
    if (!sameSize) {
        return false;
    }
    for (unsigned i = 0; i < a.size(); ++i) {
        const bool wordMatches = std::bit_cast<uint32_t>(a[i]) == std::bit_cast<uint32_t>(b[i]);
        if (!wordMatches) {
            return false;
        }
    }
    return true;
}

unsigned selectedRow(uint32_t raw)
{
    // Direct narrow profiles intentionally keep only the low source-width bits.
    if constexpr (BOUNDED_MODE == 1) {
        return static_cast<uint8_t>(raw);
    } else if constexpr (BOUNDED_MODE == 2) {
        return static_cast<uint16_t>(raw);
    }
    return raw % sourceRows;
}

int checkDataset(unsigned dataset)
{
    std::vector<float> input(sourceRows * columns + 2 * guard);
    std::vector<int32_t> words(choices + 2 * guard, std::bit_cast<int32_t>(sentinel));
    std::vector<float> output(choices * columns + 2 * guard, std::bit_cast<float>(sentinel));
    for (unsigned i = 0; i < input.size(); ++i) {
        // Defined unsigned wrap generates arbitrary payloads, including NaNs.
        const uint32_t raw = patterns[(i + dataset) % patterns.size()] ^ (i * 1664525U + dataset);
        input[i] = std::bit_cast<float>(raw);
    }
    auto expected = output;
    for (unsigned choice = 0; choice < choices; ++choice) {
        const uint32_t raw = patterns[(choice + dataset * 3) % patterns.size()];
        words[guard + choice] = std::bit_cast<int32_t>(raw);
        const unsigned row = selectedRow(raw);
        if (row >= sourceRows) {
            return 3;
        }
        for (unsigned column = 0; column < columns; ++column) {
            expected[guard + choice * columns + column] = input[guard + row * columns + column];
        }
    }
    const auto originalInput = input;
    const auto originalWords = words;
    a1_bounded_integer_dma(input.data() + guard, words.data() + guard, output.data() + guard);
    const bool inputsUnchanged = sameWords(input, originalInput) && words == originalWords;
    if (!inputsUnchanged) {
        return 1;
    }
    if (!sameWords(output, expected)) {
        std::fprintf(stderr, "bounded integer DMA mismatch, dataset %u\n", dataset);
        return 2;
    }
    return 0;
}
} // namespace

int main()
{
    for (unsigned dataset = 0; dataset < 6; ++dataset) {
        const int result = checkDataset(dataset);
        if (result != 0) {
            return result;
        }
    }
    return 0;
}
