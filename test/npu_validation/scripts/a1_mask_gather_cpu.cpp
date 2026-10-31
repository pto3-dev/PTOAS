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
#ifdef MASK_FP16
using Data = half;
using Index = half;
using Bits = uint16_t;
#else
#ifndef MASK_CTYPE
#define MASK_CTYPE float
#endif
using Data = MASK_CTYPE;
using Index = uint32_t;
using Bits = uint32_t;
#endif
constexpr unsigned sourceCount = MASK_SOURCE_COLS, selectedCount = sourceCount / MASK_FACTOR, guard = 16;
constexpr Bits sentinel = static_cast<Bits>(0xa55ac33cU);

Bits pattern(unsigned i, unsigned dataset)
{
    constexpr std::array<uint32_t, 16> values = {0U, 0x80000000U, 0x7fc01234U, 0xffc05678U,
        0x7f800001U, 0x7f800000U, 0xffffffffU, 0x80000001U, 0x01020304U, 0x00008000U,
        0x00007c00U, 0x0000fc00U, 0x00007e13U, 0x0000fe55U, 0x00000001U, 0x00007bffU};
    const uint32_t value = values[(i + dataset) % values.size()];
    return static_cast<Bits>(dataset < 2 ? value : value ^ i * 1664525U);
}

template <typename Element>
bool checkField(const std::vector<Element>& field, const std::vector<Data>& input, unsigned offset)
{
    for (unsigned i = 0; i < field.size(); ++i) {
        const bool active = i >= guard && i < guard + selectedCount;
        const Bits expected = active ? std::bit_cast<Bits>(input[guard + (i - guard) * MASK_FACTOR + offset]) :
                                       sentinel;
        const bool mismatch = std::bit_cast<Bits>(field[i]) != expected;
        if (mismatch) {
            return false;
        }
    }
    return true;
}

bool runDataset(unsigned dataset)
{
    std::vector<Data> input(sourceCount + 2 * guard), scores(selectedCount + 2 * guard, std::bit_cast<Data>(sentinel));
    std::vector<Index> indices(selectedCount + 2 * guard, std::bit_cast<Index>(sentinel));
    for (unsigned i = 0; i < input.size(); ++i) {
        input[i] = std::bit_cast<Data>(pattern(i, dataset));
    }
    const auto original = input;
    a1_mask_fields(scores.data() + guard, indices.data() + guard, input.data() + guard);
    for (unsigned i = 0; i < input.size(); ++i) {
        const bool changed = std::bit_cast<Bits>(input[i]) != std::bit_cast<Bits>(original[i]);
        if (changed) {
            return false;
        }
    }
    return checkField(scores, original, MASK_OFFSET_A) && checkField(indices, original, MASK_OFFSET_B);
}
} // namespace

int main()
{
    for (unsigned dataset = 0; dataset < 6; ++dataset) {
        if (!runDataset(dataset)) {
            std::fprintf(stderr, "mask gather dataset %u output/guard mismatch\n", dataset);
            return 1;
        }
    }
    return 0;
}
