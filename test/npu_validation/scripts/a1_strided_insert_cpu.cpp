/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/
// Appended to unmodified PTOAS output; macros describe a trusted test profile.
#include <bit>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
constexpr size_t sourceRows = PTO_TEST_ROWSTEP_SRC_ROWS;
constexpr size_t sourceColumns = PTO_TEST_ROWSTEP_SRC_COLS;
constexpr size_t targetRows = PTO_TEST_ROWSTEP_DST_ROWS;
constexpr size_t targetColumns = PTO_TEST_ROWSTEP_DST_COLS;
constexpr size_t rowStep = PTO_TEST_ROWSTEP_ROW_STEP;
constexpr size_t firstRow = PTO_TEST_ROWSTEP_ROW;
constexpr size_t firstColumn = PTO_TEST_ROWSTEP_COLUMN;
constexpr size_t copies = PTO_TEST_ROWSTEP_COPIES;
constexpr size_t groups = PTO_TEST_ROWSTEP_GROUPS;
constexpr size_t sparse = PTO_TEST_ROWSTEP_SPARSE;
constexpr size_t matrixAddress = PTO_TEST_ROWSTEP_MAT_ADDRESS;
constexpr size_t inputRows = sparse ? groups * sourceRows : copies * sourceRows;
constexpr size_t inputStride = sparse ? 9 * sourceColumns : sourceColumns + 16;
constexpr size_t matrixCount = targetRows * targetColumns;
constexpr size_t matrixBytes = matrixCount * sizeof(half);
constexpr size_t guardElements = 16;
constexpr size_t l1Bytes = 1024 * 1024;
using Matrix = pto::Tile<pto::TileType::Mat, half, targetRows, targetColumns, pto::BLayout::ColMajor,
                         targetRows, targetColumns, pto::SLayout::RowMajor, 512>;
using Guard = pto::Tile<pto::TileType::Mat, half, 16, 16, pto::BLayout::ColMajor,
                        16, 16, pto::SLayout::RowMajor, 512>;

uint16_t bits(half value) { return std::bit_cast<uint16_t>(value); }
half value(uint16_t raw) { return std::bit_cast<half>(raw); }

std::vector<half> payload(size_t count, unsigned seed, unsigned factor)
{
    std::vector<half> result(count + 2 * guardElements, value(0x3555));
    for (size_t i = 0; i < count; ++i) {
        result[guardElements + i] = value(static_cast<uint16_t>(i * factor + seed));
    }
    return result;
}

bool sameBits(const std::vector<half> &first, const std::vector<half> &second)
{
    for (size_t i = 0; i < first.size(); ++i) {
        const bool different = bits(first[i]) != bits(second[i]);
        if (different) {
            return false;
        }
    }
    return true;
}

void fillGuard(Guard &guard)
{
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
            guard.SetElement(row, col, value(0x3555));
        }
    }
}

bool intact(Guard &guard)
{
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
            const bool damaged = bits(guard.GetElement(row, col)) != 0x3555;
            if (damaged) {
                return false;
            }
        }
    }
    return true;
}

std::vector<uint16_t> reference(const std::vector<half> &input, const std::vector<half> &initial)
{
    std::vector<uint16_t> result(matrixCount, 0);
    if (sparse) {
        for (size_t i = 0; i < inputRows * inputStride; ++i) {
            const size_t channel = i / (9 * sourceColumns);
            const size_t tap = i / sourceColumns % 9;
            const size_t row = channel / sourceRows * 2304 + tap * 16 + channel % sourceRows * 145;
            result[row * targetColumns + firstColumn + i % sourceColumns] = bits(input[guardElements + i]);
        }
        return result;
    }
    for (size_t i = 0; i < matrixCount; ++i) {
        result[i] = bits(initial[guardElements + i]);
    }
    for (size_t copy = 0; copy < copies; ++copy) {
        for (size_t i = 0; i < sourceRows * sourceColumns; ++i) {
            const size_t row = firstRow + (copies > 1 ? copy : 0) + i / sourceColumns * rowStep;
            const size_t src = (copy * sourceRows + i / sourceColumns) * inputStride + i % sourceColumns;
            result[row * targetColumns + firstColumn + i % sourceColumns] = bits(input[guardElements + src]);
        }
    }
    return result;
}

bool checkResult(Matrix &matrix, const std::vector<uint16_t> &expected)
{
    for (size_t i = 0; i < matrixCount; ++i) {
        const bool different = bits(matrix.GetElement(i / targetColumns, i % targetColumns)) != expected[i];
        if (different) {
            std::cerr << "bitwise mismatch at " << i << '\n';
            return false;
        }
    }
    return true;
}

int checkDataset(unsigned dataset)
{
    auto input = payload(inputRows * inputStride, dataset * 251, 4093);
    auto initial = payload(matrixCount, dataset * 53, 277);
    const auto beforeInput = input;
    const auto beforeInitial = initial;
    Matrix matrix;
    Guard lower, upper;
    pto::TASSIGN(matrix, matrixAddress);
    if (matrixAddress >= 512) {
        pto::TASSIGN(lower, matrixAddress - 512);
        fillGuard(lower);
    }
    if (matrixAddress + matrixBytes + 512 <= l1Bytes) {
        pto::TASSIGN(upper, matrixAddress + matrixBytes);
        fillGuard(upper);
    }
    for (size_t i = 0; i < matrixCount; ++i) {
        matrix.SetElement(i / targetColumns, i % targetColumns, value(0x3555));
    }
    rowstep_kernel(input.data() + guardElements, initial.data() + guardElements);
    const bool validOutput = checkResult(matrix, reference(beforeInput, beforeInitial));
    const bool unchangedInput = sameBits(input, beforeInput) && sameBits(initial, beforeInitial);
    if (!validOutput || !unchangedInput) {
        return 1;
    }
    const bool lowerDamaged = matrixAddress >= 512 && !intact(lower);
    const bool upperDamaged = matrixAddress + matrixBytes + 512 <= l1Bytes && !intact(upper);
    if (lowerDamaged || upperDamaged) {
        return 2;
    }
    return 0;
}
} // namespace

int main()
{
    for (unsigned dataset = 0; dataset < 3; ++dataset) {
        const int result = checkDataset(dataset);
        if (result != 0) {
            return result;
        }
    }
    return 0;
}
