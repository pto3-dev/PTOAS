// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
constexpr int kGuard = 16;
constexpr int kPixels = IMG_H * IMG_W;
constexpr int kInputCount = IMG_IMAGES * kPixels * IMG_C;
constexpr int kWeightCount = IMG_K * 16;
constexpr int kOutputImages = IMG_SUM ? 1 : IMG_IMAGES;
constexpr int kOutputCount = kOutputImages * IMG_ROWS * 16;
static_assert(IMG_H > 0 && IMG_W > 0 && IMG_C > 0 && IMG_C % 16 == 0);
static_assert(IMG_SH > 0 && IMG_SW > 0 && IMG_DH > 0 && IMG_DW > 0);
static_assert(IMG_K0 >= 0 && IMG_K0 % 16 == 0 && IMG_K0 + IMG_K <= IMG_C * IMG_KH * IMG_KW);
constexpr int kOutputWidth = (IMG_W + IMG_PL + IMG_PR - IMG_DW * (IMG_KW - 1) - 1) / IMG_SW + 1;

constexpr int kOutputHeight = (IMG_H + IMG_PT + IMG_PB - IMG_DH * (IMG_KH - 1) - 1) / IMG_SH + 1;
static_assert(IMG_M0 >= 0 && IMG_M0 + IMG_ROWS <= kOutputWidth * kOutputHeight);

#if defined(IMG_FILTER_ORDER)
constexpr int kFilterOrder[] = {IMG_FILTER_ORDER};
static_assert(sizeof(kFilterOrder) / sizeof(kFilterOrder[0]) * 16 == IMG_K);
static_assert(IMG_K0 % (IMG_KH * IMG_KW * 16) == 0);
constexpr bool validFilterOrder() {
  for (int tap : kFilterOrder) {
    if (tap < 0 || tap >= IMG_KH * IMG_KW) {
      return false;
    }
  }
  return true;
}
static_assert(validFilterOrder());
#endif

int reductionIndex(int k) {
#if defined(IMG_FILTER_ORDER)
  return IMG_K0 + kFilterOrder[k / 16] * 16 + k % 16;
#else
  return IMG_K0 + k;
#endif
}

half sample(int index, int dataset, unsigned multiplier, float divisor) {
  if (dataset == 2) {
    const unsigned pattern = static_cast<unsigned>(index) * multiplier % 63488;
    const auto bits = static_cast<uint16_t>(pattern < 31744 ? pattern : (pattern - 31744) | 0x8000);
    return std::bit_cast<half>(bits);
  }
  const int numerator = (index * static_cast<int>(multiplier)) % 101 - 50;
  return half(static_cast<float>(numerator) / (dataset == 0 ? 16.0f : divisor));
}

float imageValue(const std::vector<half> &input, int image, int m, int reduction) {
  const int lane = reduction % 16;
  const int tap = reduction / 16;
  const int channel = (tap / (IMG_KH * IMG_KW)) * 16 + lane;
  const int y = m / kOutputWidth * IMG_SH + (tap / IMG_KW) % IMG_KH * IMG_DH - IMG_PT;
  const int x = m % kOutputWidth * IMG_SW + tap % IMG_KW * IMG_DW - IMG_PL;
  if (y < 0 || y >= IMG_H || x < 0 || x >= IMG_W) {
    return 0.0f;
  }
  return static_cast<float>(input[kGuard + (image * kPixels + y * IMG_W + x) * IMG_C + channel]);
}

void reference(const std::vector<half> &input, const std::vector<half> &weight, std::vector<float> &expected) {
  for (int image = 0; image < kOutputImages; ++image) {
    for (int row = 0; row < IMG_ROWS; ++row) {
      for (int column = 0; column < 16; ++column) {
        float total = 0.0f;
        for (int source = 0; source < (IMG_SUM ? IMG_IMAGES : 1); ++source) {
          const int selected = IMG_SUM ? source : image;
          for (int k = 0; k < IMG_K; ++k) {
            total = std::fma(imageValue(input, selected, IMG_M0 + row, reductionIndex(k)),
                             static_cast<float>(weight[kGuard + k * 16 + column]), total);
          }
        }
        expected[kGuard + (image * IMG_ROWS + row) * 16 + column] = total;
      }
    }
  }
}
} // namespace

int main() {
  for (int dataset = 0; dataset < 3; ++dataset) {
    std::vector<half> input(kInputCount + 2 * kGuard, half(7));
    std::vector<half> weight(kWeightCount + 2 * kGuard, half(11));
    std::vector<float> output(kOutputCount + 2 * kGuard, 9.0f);
    auto expected = output;
    for (int i = 0; i < kInputCount; ++i) {
      input[kGuard + i] = sample(i, dataset, 73, 37.0f);
    }
    for (int i = 0; i < kWeightCount; ++i) {
      weight[kGuard + i] = sample(i, dataset, 97, 43.0f);
    }
    reference(input, weight, expected);
    const auto originalInput = input;
    const auto originalWeight = weight;
    a1_img2col_mat(output.data() + kGuard, input.data() + kGuard, weight.data() + kGuard);
    // Clones have exactly the same extent. Compare the complete buffers,
    // including both guards, without any floating-point tolerance.
    const bool inputUnchanged = std::memcmp(input.data(), originalInput.data(), input.size() * sizeof(half)) == 0;
    const bool weightUnchanged = std::memcmp(weight.data(), originalWeight.data(), weight.size() * sizeof(half)) == 0;
    if (!inputUnchanged || !weightUnchanged) {
      return 1;
    }
    const bool outputMatches = std::memcmp(output.data(), expected.data(), output.size() * sizeof(float)) == 0;
    if (!outputMatches) {
      for (unsigned i = 0; i < output.size(); ++i) {
        const auto actual = std::bit_cast<uint32_t>(output[i]);
        const auto wanted = std::bit_cast<uint32_t>(expected[i]);
        if (actual != wanted) {
          std::fprintf(stderr, "Mat im2col dataset=%d index=%u actual=%08x expected=%08x\n", dataset, i,
                       actual, wanted);
          break;
        }
      }
      return 2;
    }
  }
  return 0;
}
