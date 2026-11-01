// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_SUPPORT_A1_TARGET_H
#define PTO_SUPPORT_A1_TARGET_H

#include "PTO/IR/PTO.h"
#include <cstdint>
#include <optional>

namespace mlir::pto {

// L1 uses the full 1 MiB reported by CANN Ascend310P platform configuration.
// UB retains its conservative profile budget; this does not change other
// targets or the existing alignment, lifetime and address-range checks.
struct A1MemorySpec {
    uint64_t capacityBytes;
    uint64_t alignmentBytes;
};
inline constexpr uint64_t kA1UBCapacityBytes = 128 * 1024;
inline constexpr uint64_t kA1L1CapacityBytes = 1024 * 1024;
inline constexpr unsigned kA1ReservedEventId = 7;

inline A1MemorySpec getA1MemorySpec(AddressSpace space)
{
    switch (space) {
        case AddressSpace::VEC:
            return {kA1UBCapacityBytes, 32};
        case AddressSpace::MAT:
            return {kA1L1CapacityBytes, 512};
        case AddressSpace::LEFT:
        case AddressSpace::RIGHT:
            return {64 * 1024, 512};
        case AddressSpace::ACC:
            return {256 * 1024, 512};
        default:
            return {0, 1};
    }
}

// Prove the target-specific capacity before multiplying external dimensions.
inline std::optional<uint64_t> getA1StaticTileBytes(int64_t rows, int64_t columns,
                                                 unsigned elementBytes, AddressSpace space)
{
    if (rows <= 0 || columns <= 0 || elementBytes == 0) {
        return std::nullopt;
    }
    const uint64_t maxElements = getA1MemorySpec(space).capacityBytes / elementBytes;
    const uint64_t width = static_cast<uint64_t>(columns);
    if (width > maxElements) {
        return std::nullopt;
    }
    const uint64_t maxRows = maxElements / width;
    const uint64_t height = static_cast<uint64_t>(rows);
    if (height > maxRows) {
        return std::nullopt;
    }
    return height * width * elementBytes;
}

// A source C16 row writes one 32-byte destination block. The DMA gap
// field is 16 bits, so the maximum logical row step is one greater.
inline constexpr int64_t kA1MaxInsertionRowStep = 65536;
inline std::optional<int64_t> getA1StridedInsertSpan(int64_t sourceRows, int64_t destinationRows,
                                                 int64_t rowStep)
{
    if (sourceRows <= 0 || destinationRows <= 0 || rowStep <= 0 || rowStep > kA1MaxInsertionRowStep) {
        return std::nullopt;
    }
    // Prove the product bounded before multiplying; malformed IR may carry
    // arbitrary signed 64-bit dimensions or attributes.
    const int64_t maxSourceRowsMinusOne = (destinationRows - 1) / rowStep;
    if (sourceRows - 1 > maxSourceRowsMinusOne) {
        return std::nullopt;
    }
    return (sourceRows - 1) * rowStep + 1;
}

// Materialize full valid shapes only when both allocation operands prove them.
void normalizeA1StaticValidShapes(ModuleOp module);
LogicalResult validateA1Capabilities(ModuleOp module);

} // namespace mlir::pto
#endif // PTO_SUPPORT_A1_TARGET_H
