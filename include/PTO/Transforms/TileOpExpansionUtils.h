// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_TRANSFORMS_TILEOPEXPANSIONUTILS_H
#define PTO_TRANSFORMS_TILEOPEXPANSIONUTILS_H

#include "PTO/IR/PTO.h"

namespace mlir::pto {

/// Return whether an operation is a TileLib template expansion candidate.
/// Frontend pipe/sync pseudo-ops use TileOpInterface for surface
/// classification but must be handled by their dedicated lowering instead.
inline bool isTileLibExpandableOp(Operation *op) {
  if (!op || !isa<TileOpInterface>(op)) {
    return false;
  }
  // Row-scatter is an A1 native ISA form. Rectangular TileLib templates
  // cannot represent its sparse destination window.
  if (auto insert = dyn_cast<TInsertOp>(op)) {
    if (insert.getRowStepAttr()) {
      return false;
    }
  }
  // C16 block selection is native MTE1 DMA, not a rectangular TileLib slice.
  if (auto extract = dyn_cast<TExtractOp>(op)) {
    const bool nativeExtraction = extract.getBlockStepAttr() || isTargetArchA1(op);
    if (nativeExtraction) {
      return false;
    }
  }
  return !isa<TReshapeOp, TSyncOp, TAllocToAivOp, TAllocToAicOp,
              TPushToAivOp, TPushToAicOp, TPopFromAicOp, TPopFromAivOp,
              TFreeFromAicOp, TFreeFromAivOp>(op);
}

} // namespace mlir::pto

#endif // PTO_TRANSFORMS_TILEOPEXPANSIONUTILS_H
