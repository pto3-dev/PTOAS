/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Appended to unmodified generated C++; compare raw payloads and consumed counts.
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
constexpr unsigned guard = 16;
constexpr uint32_t sentinel = 0xa55ac33cU;
constexpr std::array<unsigned, 4> columns = {MERGE_COLS0, MERGE_COLS1, MERGE_COLS2, MERGE_COLS3};
constexpr unsigned ways = MERGE_WAYS, outputCols = MERGE_OUTPUT_COLS;
using Inputs = std::array<std::vector<float>, 4>;
struct Pair {
    float score;
    uint32_t payload;
    unsigned source;
};

float scoreFor(unsigned dataset, unsigned source, unsigned pair)
{
    constexpr std::array<uint32_t, 8> patterns = {
        0U, 0x80000000U, 0x7f800000U, 0xff800000U, 0x3f800000U, 0xbf800000U, 1U, 0x80000001U};
    switch (dataset) {
        case 0: return static_cast<float>((pair * 17 + source * 13) % 97) - 48.0f;
        case 1: return 0.5f;
        case 2: return std::bit_cast<float>(patterns[(pair + source) % patterns.size()]);
        case 3: return static_cast<float>(10000 * source + pair);
        case 4: return static_cast<float>(10000 * (ways - source) + pair);
        default: return static_cast<float>((pair / 3 + source) % 7);
    }
}

std::vector<Pair> initializeInputs(Inputs& inputs, unsigned dataset)
{
    std::vector<Pair> all;
    const unsigned stride = *std::max_element(columns.begin(), columns.end()) + 32;
    for (unsigned source = 0; source < ways; ++source) {
        inputs[source].assign(stride + 2 * guard, std::bit_cast<float>(sentinel));
        std::vector<Pair> list;
        for (unsigned pair = 0; pair < columns[source] / 2; ++pair) {
            list.push_back({scoreFor(dataset, source, pair), 0x7fc00000U ^ (source * 65537U + pair * 257U), source});
        }
        std::stable_sort(list.begin(), list.end(), [](const Pair& a, const Pair& b) { return a.score > b.score; });
        for (unsigned pair = 0; pair < list.size(); ++pair) {
            inputs[source][guard + 16 + pair * 2] = list[pair].score;
            inputs[source][guard + 17 + pair * 2] = std::bit_cast<float>(list[pair].payload);
        }
        all.insert(all.end(), list.begin(), list.end());
    }
    std::stable_sort(all.begin(), all.end(), [](const Pair& a, const Pair& b) { return a.score > b.score; });
    return all;
}

std::array<unsigned, 4> reference(const std::vector<Pair>& sorted, std::vector<float>& expected)
{
    std::array<unsigned, 4> consumed{};
    for (unsigned pair = 0; pair < outputCols / 2; ++pair) {
        const auto item = sorted[pair];
        expected[guard + 16 + 2 * pair] = item.score;
        expected[guard + 17 + 2 * pair] = std::bit_cast<float>(item.payload);
        ++consumed[item.source];
        if (MERGE_EXHAUSTED && consumed[item.source] == columns[item.source] / 2) {
            break;
        }
    }
    return consumed;
}

bool equalBits(const std::vector<float>& actual, const std::vector<float>& expected)
{
    for (unsigned i = 0; i < actual.size(); ++i) {
        const bool differs = std::bit_cast<uint32_t>(actual[i]) != std::bit_cast<uint32_t>(expected[i]);
        if (differs) {
            std::fprintf(stderr, "merge bit mismatch at element %u\n", i);
            return false;
        }
    }
    return true;
}

bool checkDataset(unsigned dataset)
{
    Inputs inputs;
    const auto sorted = initializeInputs(inputs, dataset);
    const auto original = inputs;
    std::vector<float> output(outputCols + 32 + 2 * guard, std::bit_cast<float>(sentinel));
    // An exhausting merge must retain real initialized suffix values, not zeroes.
    for (unsigned i = 0; i < outputCols; ++i) {
        output[guard + 16 + i] = std::bit_cast<float>(0x3f000000U + i);
    }
    auto expected = output;
    const auto consumed = reference(sorted, expected);
    const auto actual = a1_mrgsort(output.data() + guard, inputs[0].data() + guard, inputs[1].data() + guard,
                                   ways > 2 ? inputs[2].data() + guard : nullptr,
                                   ways > 3 ? inputs[3].data() + guard : nullptr);
    const std::array<unsigned, 4> counts = {
        actual.mrgSortList0, actual.mrgSortList1, actual.mrgSortList2, actual.mrgSortList3};
    if (counts != consumed || !equalBits(output, expected)) {
        std::fprintf(stderr, "merge output/count mismatch, dataset %u\n", dataset);
        return false;
    }
    for (unsigned source = 0; source < ways; ++source) {
        if (!equalBits(inputs[source], original[source])) {
            return false;
        }
    }
    return true;
}
} // namespace

int main()
{
    for (unsigned dataset = 0; dataset < 6; ++dataset) {
        if (!checkDataset(dataset)) {
            return 1;
        }
    }
    return 0;
}
