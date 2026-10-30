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
#include <cstdint>
#include <cstring>
#include <vector>

#ifdef MULS_FP16
using Element = half;
constexpr unsigned Datasets = 1024;
#else
using Element = float;
constexpr unsigned Datasets = 4;
#endif
constexpr unsigned Count = 64, Guard = 16;

int main()
{
    const std::array<float, 6> factors = {0.1767766922712326f, -2.0f, 0.0f, -0.0f, 0x1p-30f, 1.0003f};
    for (float factor : factors) {
        const Element scalar = static_cast<Element>(factor);
        for (unsigned dataset = 0; dataset < Datasets; ++dataset) {
            for (bool alias : {false, true}) {
                std::vector<Element> input(Count + 2 * Guard, Element(9));
                std::vector<Element> output(Count + 2 * Guard, Element(9));
                for (unsigned i = 0; i < Count; ++i) {
#ifdef MULS_FP16
                    // Visit every finite FP16 encoding, including both signed zeros.
                    uint16_t bits = static_cast<uint16_t>(dataset * Count + i);
                    if ((bits & 0x7c00) == 0x7c00)
                        bits = 0;
                    std::memcpy(&input[Guard + i], &bits, sizeof(bits));
#else
                    input[Guard + i] = Element((static_cast<int>((i * 17 + dataset * 7) % 97) - 48) / 31.0f);
                    if (i < 4)
                        input[Guard + i] = i % 2 ? Element(-0.0f) : Element(0.0f);
#endif
                }
                const auto original = input;
                auto* result = alias ? input.data() : output.data();
                KERNEL(result + Guard, input.data() + Guard, scalar);
                for (unsigned i = 0; i < Count; ++i) {
                    const Element expected =
                        Element(static_cast<float>(original[Guard + i]) * static_cast<float>(scalar));
                    if (std::memcmp(&result[Guard + i], &expected, sizeof(Element)))
                        return 1;
                }
                for (unsigned i = 0; i < Guard; ++i) {
                    if (result[i] != Element(9) || result[Guard + Count + i] != Element(9))
                        return 2;
                }
                if (!alias && std::memcmp(input.data(), original.data(), input.size() * sizeof(Element)))
                    return 3;
            }
        }
    }
    return 0;
}
