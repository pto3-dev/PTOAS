/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Appended to generated PTOAS C++; KERNEL and MULS_FP16 select the case.
#include <array>
#include <bit>
#include <type_traits>
#include <cstring>
#include <vector>

using Data = GATHER_DATA;
using Index = GATHER_INDEX;
using Bits = std::conditional_t<sizeof(Data) == 2, uint16_t, uint32_t>;
static_assert(sizeof(Data) == 2 || sizeof(Data) == 4);
int main()
{
    constexpr unsigned rows = 8, sourceCols = 80, columns = 96;
    constexpr unsigned inputStride = 112, outputStride = 128, guard = 16;
    constexpr uint32_t sentinel = 0xa55ac33cU;
    constexpr std::array<uint32_t, 14> patterns = {
        0U, 0x80000000U, 0x7fc01234U, 0xffc05678U, 0x7f800001U, 0x7f800000U, 0xffffffffU,
        0x7fffffffU, 0x80000001U, 0x01020304U, 0x7e017c01U, 0xfc017e55U, 0x00008000U, 0x00003c00U};
    for (unsigned dataset = 0; dataset < 4; ++dataset) {
        std::vector<Data> input(rows * inputStride + 2 * guard), output(rows * outputStride + 2 * guard);
        std::vector<Index> indices(rows * inputStride + 2 * guard, Index(7));
        for (unsigned i = 0; i < input.size(); ++i) {
            const uint32_t bits = patterns[(i + dataset) % patterns.size()] ^ (dataset == 3 ? i * 1664525U : 0);
            input[i] = std::bit_cast<Data>(static_cast<Bits>(bits));
        }
        for (auto& value : output) {
            value = std::bit_cast<Data>(static_cast<Bits>(sentinel));
        }
        for (unsigned row = 0; row < rows; ++row)
            for (unsigned col = 0; col < columns; ++col) {
                const unsigned i = row * columns + col, sourceSize = rows * sourceCols;
                const unsigned index = dataset == 0 ? i % sourceSize : dataset == 1 ? sourceSize - 1 - i % sourceSize :
                    dataset == 2 ? (i % 7 == 0 ? 0 : sourceSize - 1) : (i * 37 + 19) % sourceSize;
                indices[guard + row * inputStride + 8 + col] = static_cast<Index>(index);
            }
        const auto original = input;
        const auto originalIndices = indices;
        a1_indexed_gather(output.data() + guard, input.data() + guard, indices.data() + guard);
        const bool inputChanged = std::memcmp(input.data(), original.data(), input.size() * sizeof(Data)) ||
                                  indices != originalIndices;
        if (inputChanged) {
            return 1;
        }
        for (unsigned i = 0; i < output.size(); ++i) {
            const void* expected = &sentinel;
            if (i >= guard && i < guard + rows * outputStride) {
                const unsigned row = (i - guard) / outputStride, col = (i - guard) % outputStride;
                if (col >= 16 && col < 16 + columns) {
                    const unsigned index = originalIndices[guard + row * inputStride + 8 + col - 16];
                    expected = &original[guard + (index / sourceCols) * inputStride + 16 + index % sourceCols];
                }
            }
            const bool bitsDiffer = std::memcmp(&output[i], expected, sizeof(Data));
            if (bitsDiffer) {
                return 2;
            }
        }
    }
    return 0;
}
