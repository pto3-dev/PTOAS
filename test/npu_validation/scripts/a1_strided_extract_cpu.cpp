/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Appended to unmodified compiler output; definitions describe trusted profiles.
#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
constexpr size_t sourceRows = PTO_TEST_EXTRACT_SOURCE_ROWS, sourceColumns = PTO_TEST_EXTRACT_SOURCE_COLS;
constexpr size_t targetRows = PTO_TEST_EXTRACT_TARGET_ROWS, targetColumns = PTO_TEST_EXTRACT_TARGET_COLS;
constexpr size_t sourceAddress = PTO_TEST_EXTRACT_SOURCE_ADDRESS;
constexpr size_t right0Address = PTO_TEST_EXTRACT_RIGHT0_ADDRESS, right1Address = PTO_TEST_EXTRACT_RIGHT1_ADDRESS;
constexpr size_t channels = PTO_TEST_EXTRACT_CHANNELS, groups = PTO_TEST_EXTRACT_GROUPS;
constexpr size_t rowStep = PTO_TEST_EXTRACT_STEP, originRow = PTO_TEST_EXTRACT_ROW,
                 originColumn = PTO_TEST_EXTRACT_COLUMN;
constexpr size_t leftMatAddress = PTO_TEST_EXTRACT_LEFT_MAT_ADDRESS;
constexpr size_t guardElements = 16;
constexpr size_t inputRows = channels ? channels * groups : sourceRows;
constexpr size_t inputStride = channels ? 9 * sourceColumns : sourceColumns + 16;
constexpr auto sourceLayout = PTO_TEST_EXTRACT_ZZ ? pto::BLayout::RowMajor : pto::BLayout::ColMajor;
using Resident = pto::Tile<
    pto::TileType::Mat, half, sourceRows, sourceColumns, sourceLayout, sourceRows, sourceColumns,
    pto::SLayout::RowMajor, 512>;
using Right = pto::TileRight<half, targetRows, targetColumns>;
using Acc = pto::TileAcc<float, 16, targetColumns>;
using MatGuard =
    pto::Tile<pto::TileType::Mat, half, 16, 16, pto::BLayout::ColMajor, 16, 16, pto::SLayout::RowMajor, 512>;
using RightGuard = pto::TileRight<half, 16, 16>;
uint16_t bits(half value) { return std::bit_cast<uint16_t>(value); }
half fromBits(uint16_t value) { return std::bit_cast<half>(value); }

bool sameBits(const std::vector<half>& first, const std::vector<half>& second)
{
    const bool sameSize = first.size() == second.size();
    if (!sameSize) {
        return false;
    }
    for (size_t index = 0; index < first.size(); ++index) {
        const bool same = bits(first[index]) == bits(second[index]);
        if (!same) {
            return false;
        }
    }
    return true;
}

template <typename Guard>
void paintGuard(Guard& guard)
{
    for (size_t row = 0; row < 16; ++row) {
        for (size_t column = 0; column < 16; ++column) {
            guard.SetElement(row, column, fromBits(0x3555));
        }
    }
}

template <typename Guard>
bool intact(Guard& guard)
{
    for (size_t row = 0; row < 16; ++row) {
        for (size_t column = 0; column < 16; ++column) {
            const bool damaged = bits(guard.GetElement(row, column)) != 0x3555;
            if (damaged) {
                return false;
            }
        }
    }
    return true;
}

// Interpret compact GM weights directly; never use generated L1 as the oracle.
half weight(const std::vector<half>& input, size_t row, size_t column)
{
    if constexpr (channels == 0) {
        return input[guardElements + row * inputStride + 16 + column];
    }
    const size_t group = row / 2304, local = row % 2304;
    for (size_t channel = 0; channel < channels; ++channel) {
        if (local >= channel * 145) {
            const size_t delta = local - channel * 145;
            if (delta % 16 == 0 && delta / 16 < 9) {
                return input
                    [guardElements + (group * channels + channel) * inputStride + delta / 16 * sourceColumns + column];
            }
        }
    }
    return half(0);
}

bool checkSource(Resident& source, const std::vector<half>& input)
{
    for (size_t row = 0; row < sourceRows; ++row) {
        for (size_t column = 0; column < sourceColumns; ++column) {
            const bool matches = bits(source.GetElement(row, column)) == bits(weight(input, row, column));
            if (!matches) {
                std::cerr << "resident mismatch " << row << ',' << column << '\n';
                return false;
            }
        }
    }
    return true;
}

bool checkRight(Right& right, const std::vector<half>& input, size_t rowOrigin, size_t columnOrigin)
{
    for (size_t row = 0; row < targetRows; ++row) {
        const size_t selectedRow = rowOrigin + (row / 16) * rowStep * 16 + row % 16;
        for (size_t column = 0; column < targetColumns; ++column) {
            const auto actual = bits(right.GetElement(row, column));
            const auto expected = bits(weight(input, selectedRow, columnOrigin + column));
            if (actual != expected) {
                std::cerr << "Right mismatch " << row << ',' << column << '\n';
                return false;
            }
        }
    }
    return true;
}

bool checkCube(const std::vector<half>& input, const std::vector<half>& lhs)
{
    Acc acc;
    pto::TASSIGN<PTO_TEST_EXTRACT_ACC_ADDRESS>(acc);
    for (size_t row = 0; row < 16; ++row) {
        for (size_t column = 0; column < targetColumns; ++column) {
            float expected = 0;
            constexpr std::array<size_t, 5> origins = {0, 16, 32, 48, (groups - 1) * 2304 + 576};
            for (size_t start : origins) {
                for (size_t k = 0; k < 144; ++k) {
                    const size_t sourceRow = start + (k / 16) * 64 + k % 16;
                    expected += float(lhs[guardElements + row * 144 + k]) * float(weight(input, sourceRow, column));
                }
            }
            const auto actualBits = std::bit_cast<uint32_t>(acc.GetElement(row, column));
            const auto expectedBits = std::bit_cast<uint32_t>(expected);
            if (actualBits != expectedBits) {
                std::cerr << "Cube mismatch " << row << ',' << column << '\n';
                return false;
            }
        }
    }
    return true;
}

constexpr bool spareMatGuard(size_t address)
{
    return channels == 0 || address + 512 <= leftMatAddress || address >= leftMatAddress + 16 * 144 * 2;
}

bool checkDataset(unsigned dataset)
{
    std::vector<half> input(inputRows * inputStride + 2 * guardElements, fromBits(0x3555));
    std::vector<half> lhs(16 * 144 + 2 * guardElements, fromBits(0x3555));
    for (size_t index = 0; index < inputRows * inputStride; ++index) {
        input[guardElements + index] = channels ? half(int((index * 5 + dataset) % 7) - 3) :
                                                  fromBits(static_cast<uint16_t>(index * 4093 + dataset * 73));
    }
    for (size_t index = 0; index < 16 * 144; ++index) {
        lhs[guardElements + index] = half(int((index * 3 + dataset) % 5) - 2);
    }
    const auto originalInput = input, originalLhs = lhs;
    Resident source;
    pto::TASSIGN<PTO_TEST_EXTRACT_SOURCE_ADDRESS>(source);
    MatGuard lowerMat, upperMat;
    RightGuard lowerRight, upperRight;
    constexpr size_t matrixEnd = sourceAddress + sourceRows * sourceColumns * 2;
    constexpr size_t rightBegin = right0Address < right1Address ? right0Address : right1Address;
    constexpr size_t rightEnd =
        (right0Address > right1Address ? right0Address : right1Address) + targetRows * targetColumns * 2;
    if constexpr (sourceAddress >= 512 && spareMatGuard(sourceAddress - 512)) {
        pto::TASSIGN<sourceAddress - 512>(lowerMat);
        paintGuard(lowerMat);
    }
    if constexpr (matrixEnd + 512 <= 1048576 && spareMatGuard(matrixEnd)) {
        pto::TASSIGN<matrixEnd>(upperMat);
        paintGuard(upperMat);
    }
    if constexpr (rightBegin >= 512) {
        pto::TASSIGN<rightBegin - 512>(lowerRight);
        paintGuard(lowerRight);
    }
    if constexpr (rightEnd + 512 <= 65536) {
        pto::TASSIGN<rightEnd>(upperRight);
        paintGuard(upperRight);
    }
#if PTO_TEST_EXTRACT_CHANNELS
    rowstep_kernel(input.data() + guardElements, lhs.data() + guardElements);
#else
    extract_kernel(input.data() + guardElements, lhs.data() + guardElements);
#endif
    const bool preserved = sameBits(input, originalInput) && sameBits(lhs, originalLhs);
    const bool sourceMatches = checkSource(source, originalInput);
    if (!preserved || !sourceMatches) {
        return false;
    }
    Right right0, right1;
    pto::TASSIGN<PTO_TEST_EXTRACT_RIGHT0_ADDRESS>(right0);
    pto::TASSIGN<PTO_TEST_EXTRACT_RIGHT1_ADDRESS>(right1);
    if constexpr (channels > 0) {
        const bool firstMatches = checkRight(right0, originalInput, (groups - 1) * 2304 + 576, 0);
        const bool secondMatches = checkRight(right1, originalInput, 48, 0);
        const bool cubeMatches = checkCube(originalInput, originalLhs);
        if (!firstMatches || !secondMatches || !cubeMatches) {
            return false;
        }
    } else if (!checkRight(right0, originalInput, originRow, originColumn)) {
        return false;
    }
    if constexpr (sourceAddress >= 512 && spareMatGuard(sourceAddress - 512)) {
        if (!intact(lowerMat)) {
            return false;
        }
    }
    if constexpr (matrixEnd + 512 <= 1048576 && spareMatGuard(matrixEnd)) {
        if (!intact(upperMat)) {
            return false;
        }
    }
    if constexpr (rightBegin >= 512) {
        if (!intact(lowerRight)) {
            return false;
        }
    }
    if constexpr (rightEnd + 512 <= 65536) {
        if (!intact(upperRight)) {
            return false;
        }
    }
    return true;
}
} // namespace

int main()
{
    for (unsigned dataset = 0; dataset < 3; ++dataset) {
        if (!checkDataset(dataset)) {
            return 1;
        }
    }
    std::cout << "PASS exact outputs, resident source and readonly GM/guards\n";
    return 0;
}
