/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Appended to generated PTOAS C++; RESHAPE_CTYPE selects signed or unsigned storage.
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>

int main()
{
    using Data = RESHAPE_CTYPE;
    constexpr unsigned count = 256, guard = 16;
    constexpr uint32_t sentinel = 0xa55ac33cU;
    constexpr std::array<uint32_t, 8> patterns = {
        0U, 0x80000000U, 0x7fc01234U, 0xffc05678U, 0x7f800001U, 0xffffffffU, 0x7fffffffU, 0x01020304U};
    for (unsigned dataset = 0; dataset < 6; ++dataset) {
        std::vector<Data> input(count + 2 * guard), replacement(count + 2 * guard);
        std::vector<Data> output(2 * count + 2 * guard, std::bit_cast<Data>(sentinel));
        for (unsigned i = 0; i < input.size(); ++i) {
            input[i] = std::bit_cast<Data>(patterns[(i + dataset) % patterns.size()] ^ i * 1664525U);
            replacement[i] = std::bit_cast<Data>(patterns[(i * 7 + dataset) % patterns.size()] ^ i * 22695477U);
        }
        const auto original = input, savedReplacement = replacement;
        a1_integer_reshape(output.data() + guard, input.data() + guard, replacement.data() + guard);
        if (input != original || replacement != savedReplacement) {
            return 1;
        }
        for (unsigned i = 0; i < output.size(); ++i) {
            const uint32_t expected = i < guard || i >= output.size() - guard ? sentinel :
                i < count + guard ? std::bit_cast<uint32_t>(original[i]) :
                                    std::bit_cast<uint32_t>(savedReplacement[i - count]);
            const bool matches = std::bit_cast<uint32_t>(output[i]) == expected;
            if (!matches) {
                std::fprintf(stderr, "integer reshape dataset %u element %u mismatch\n", dataset, i);
                return 2;
            }
        }
    }
    return 0;
}
