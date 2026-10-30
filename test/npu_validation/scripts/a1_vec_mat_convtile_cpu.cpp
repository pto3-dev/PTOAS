/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Appended to generated PTOAS C++; SIDE, CHANNELS and KERNEL select the case.
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

int main()
{
    constexpr int side = SIDE, channels = CHANNELS, outputs = side * side * 16;
    constexpr int inputCount = side * side * channels, weightCount = 9 * channels * 16;
    for (int dataset = 0; dataset < 2; ++dataset) {
        std::vector<half> input(inputCount + 32, half(7));
        std::vector<half> rawWeight(weightCount), weight(weightCount + 32, half(7));
        std::vector<float> output(outputs + 32, 7.0f);
        for (int i = 0; i < inputCount; ++i)
            input[16 + i] = half(((i * 7 + dataset * 3) % 19 - 9) / 16.0f);
        for (int i = 0; i < weightCount; ++i)
            rawWeight[i] = half(((i * 5 + dataset * 11) % 17 - 8) / 32.0f);
        // The independent reference uses unpacked [FY, FX, C, N] weights.
        // Only device input packing uses [C1, FY, FX, C0, N].
        for (int fy = 0; fy < 3; ++fy)
            for (int fx = 0; fx < 3; ++fx)
                for (int c = 0; c < channels; ++c)
                    for (int n = 0; n < 16; ++n) {
                        const int packed = (((c / 16) * 9 + fy * 3 + fx) * 16 + c % 16) * 16 + n;
                        weight[16 + packed] = rawWeight[((fy * 3 + fx) * channels + c) * 16 + n];
                    }
        const auto originalInput = input;
        const auto originalWeight = weight;
        KERNEL(output.data() + 16, input.data() + 16, weight.data() + 16);
        for (int m = 0; m < side * side; ++m)
            for (int n = 0; n < 16; ++n) {
                float expected = 0.0f;
                for (int c1 = 0; c1 < channels / 16; ++c1)
                    for (int fy = 0; fy < 3; ++fy)
                        for (int fx = 0; fx < 3; ++fx)
                            for (int c0 = 0; c0 < 16; ++c0) {
                                const int y = m / side + fy - 1, x = m % side + fx - 1;
                                const int c = c1 * 16 + c0;
                                if (y >= 0 && y < side && x >= 0 && x < side)
                                    expected = std::fma(
                                        float(input[16 + (y * side + x) * channels + c]),
                                        float(rawWeight[((fy * 3 + fx) * channels + c) * 16 + n]), expected);
                            }
                if (std::bit_cast<uint32_t>(output[16 + m * 16 + n]) != std::bit_cast<uint32_t>(expected)) {
                    std::cerr << "dataset=" << dataset << " mismatch at " << m << ',' << n << '\n';
                    return 1;
                }
            }
        for (int i = 0; i < 16; ++i)
            if (output[i] != 7.0f || output[16 + outputs + i] != 7.0f)
                return 2;
        if (std::memcmp(input.data(), originalInput.data(), input.size() * sizeof(half)) ||
            std::memcmp(weight.data(), originalWeight.data(), weight.size() * sizeof(half)))
            return 3;
    }
    return 0;
}
