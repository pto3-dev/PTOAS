/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Appended to the unmodified generated kernel. The caller supplies geometry.
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct SortPair {
    float score;
    uint32_t index;
};

int main()
{
    constexpr unsigned rows = SORT_ROWS, columns = SORT_COLS, guard = 16;
    constexpr unsigned inputStride = columns + 16, outputStride = 2 * columns + 32;
    constexpr uint32_t sentinel = 0xa55ac33cU;
    constexpr std::array<uint32_t, 8> patterns = {
        0U, 0x80000000U, 0x7f800000U, 0xff800000U, 0x3f800000U, 0xbf800000U, 0x00000001U, 0x80000001U};
    for (unsigned dataset = 0; dataset < 3; ++dataset) {
        std::vector<float> input(rows * inputStride + 2 * guard);
        std::vector<float> output(rows * outputStride + 2 * guard, std::bit_cast<float>(sentinel));
        std::vector<uint32_t> indices(rows * inputStride + 2 * guard, sentinel);
        for (unsigned row = 0; row < rows; ++row) {
            for (unsigned col = 0; col < columns; ++col) {
                const float score = dataset == 0 ? static_cast<float>((col * 17 + row * 3) % 23) - 11.0f :
                    dataset == 1 ? 0.5f : std::bit_cast<float>(patterns[(col + row) % patterns.size()]);
                input[guard + row * inputStride + 16 + col] = score;
                indices[guard + row * inputStride + 8 + col] = (col * 2654435761U) ^ (row * 0x9e3779b9U);
            }
        }
        const auto original = input;
        const auto originalIndices = indices;
        auto expected = output;
        for (unsigned row = 0; row < rows; ++row) {
            for (unsigned first = 0; first < columns; first += 32) {
                std::array<SortPair, 32> pairs;
                for (unsigned j = 0; j < pairs.size(); ++j) {
                    pairs[j] = {input[guard + row * inputStride + 16 + first + j],
                                indices[guard + row * inputStride + 8 + first + j]};
                }
                std::stable_sort(pairs.begin(), pairs.end(), [](const SortPair& a, const SortPair& b) {
                    return a.score != b.score ? a.score > b.score : a.index < b.index;
                });
                for (unsigned j = 0; j < pairs.size(); ++j) {
                    const unsigned offset = guard + row * outputStride + 16 + 2 * (first + j);
                    expected[offset] = pairs[j].score;
                    expected[offset + 1] = std::bit_cast<float>(pairs[j].index);
                }
            }
        }
        a1_sort32(output.data() + guard, input.data() + guard, indices.data() + guard);
        const bool inputsChanged = std::memcmp(input.data(), original.data(), input.size() * sizeof(float)) ||
                                   indices != originalIndices;
        if (inputsChanged) {
            std::fprintf(stderr, "sort32 modified a read-only input\n");
            return 1;
        }
        for (unsigned i = 0; i < output.size(); ++i) {
            const bool bitsDiffer = std::bit_cast<uint32_t>(output[i]) != std::bit_cast<uint32_t>(expected[i]);
            if (bitsDiffer) {
                std::fprintf(stderr, "sort32 mismatch: dataset=%u element=%u\n", dataset, i);
                return 2;
            }
        }
    }
    return 0;
}
