/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

// PTOAS-generated kernel is supplied with -include, to test the real output.
int main(int argc, char** argv)
{
    if (argc != 2) return 10;
    const int cores = std::atoi(argv[1]);
    if (cores <= 0) return 11;
    constexpr int n = 400, k = 32, guard = 16;
    for (int dataset : {0, 1}) {
        std::vector<half> query(n * k), key(n * k), output(n * n + 2 * guard, half(-123.0f));
        for (int i = 0; i < n * k; ++i) {
            if (dataset == 0) {
                query[i] = half(((i * 11 + i / 32) % 31 - 15) / 16.0f);
                key[i] = half(((i * 7 + i / 32) % 29 - 14) / 16.0f);
            } else {
                const uint16_t qb = uint16_t(0x2c00 + ((i * 73 + i / 32) % 0xc00)) |
                                    ((i % 3 == 0) ? uint16_t(0x8000) : uint16_t(0));
                const uint16_t kb = uint16_t(0x2800 + ((i * 97 + i / 32) % 0x1000)) |
                                    ((i % 5 < 2) ? uint16_t(0x8000) : uint16_t(0));
                std::memcpy(&query[i], &qb, sizeof(qb));
                std::memcpy(&key[i], &kb, sizeof(kb));
            }
        }
        const auto originalQuery = query, originalKey = key;
        for (int core = 0; core < cores; ++core) {
            pto::cpu_sim::set_execution_context(core, 0, 1);
            kernel(query.data(), key.data(), output.data() + guard, core, cores);
        }
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                float sum = 0.0f;
                for (int j = 0; j < k; ++j)
                    sum = std::fma(float(query[row * k + j]), float(key[col * k + j]), sum);
                const half expected = half(sum);
                if (std::memcmp(&output[guard + row * n + col], &expected, sizeof(half)) != 0) return 1;
            }
        }
        for (int i = 0; i < guard; ++i)
            if (float(output[i]) != -123.0f || float(output[guard + n * n + i]) != -123.0f) return 2;
        if (std::memcmp(query.data(), originalQuery.data(), query.size() * sizeof(half)) != 0 ||
            std::memcmp(key.data(), originalKey.data(), key.size() * sizeof(half)) != 0) return 3;
    }
    return 0;
}
