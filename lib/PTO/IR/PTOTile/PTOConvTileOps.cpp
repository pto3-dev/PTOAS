// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

//===----------------------------------------------------------------------===//
// ConvTile / IMG2COL config op verification
//===----------------------------------------------------------------------===//

static LogicalResult verifyConvTileCommon(Operation *op, Type ty, StringRef name,
                                          bool allowLowPrecision) {
  auto ct = dyn_cast<pto::ConvTileBufType>(ty);
  if (!ct) {
    return op->emitOpError() << "expects " << name << " to be a !pto.conv_tile";
  }

  auto as = getPTOMemorySpaceEnum(ty);
  if (!as || *as != pto::AddressSpace::MAT) {
    return op->emitOpError() << "expects " << name
                             << " to be in the mat address space";
  }

  if (!allowLowPrecision && isPTOLowPrecisionType(ct.getElementType())) {
    return op->emitOpError() << name << ": dtype " << ct.getElementType()
                             << " is not supported by this op yet";
  }

  auto shape = getShapeVec(ty);
  if (shape.empty() || shape.size() > 6) {
    return op->emitOpError() << "expects " << name
                             << " to have a rank in [1, 6]";
  }
  for (unsigned i = 0; i < shape.size(); ++i) {
    if (shape[i] <= 0) {
      return op->emitOpError() << "expects " << name << " shape[" << i
                               << "] to be positive";
    }
  }

  auto bufferSize = dyn_cast<IntegerAttr>(ct.getBufferSize());
  if (!bufferSize) {
    return op->emitOpError() << "expects " << name
                             << " to have a signless integer buffer_size";
  }
  if (bufferSize.getInt() <= 0) {
    return op->emitOpError() << "expects " << name
                             << " buffer_size to be positive";
  }

  int64_t logicalSize = 1;
  for (int64_t dim : shape) {
    if (logicalSize > std::numeric_limits<int64_t>::max() / dim) {
      return op->emitOpError() << "cannot compute logical size of " << name
                               << " without overflow";
    }
    logicalSize *= dim;
  }
  if (bufferSize.getInt() < logicalSize) {
    return op->emitOpError() << "expects " << name
                             << " buffer_size to cover at least "
                             << logicalSize << " elements";
  }

  return success();
}

static LogicalResult verifyTileLikeCommon(Operation *op, Type ty, StringRef name,
                                          bool allowLowPrecision) {
  if (isa<pto::TileBufType>(ty)) {
    return verifyTileBufCommon(op, ty, name, allowLowPrecision);
  }
  if (isa<pto::ConvTileBufType>(ty)) {
    return verifyConvTileCommon(op, ty, name, allowLowPrecision);
  }
  return op->emitOpError() << "expects " << name
                           << " to be a !pto.tile_buf or !pto.conv_tile";
}

// Shared verifier for the fmatrix-programming ops on a conv_tile source
// (set_fmatrix, set_img2col_rpt, set_img2col_padding): all require a valid
// mat conv_tile src and a manual fmatrix_mode selector.
template <typename OpT>
static LogicalResult verifySetFmatrixFamilyOp(OpT op) {
  if (failed(verifyConvTileCommon(op, op.getSrc().getType(), "src",
                                  /*allowLowPrecision=*/true))) {
    return failure();
  }
  auto mode = op.getFmatrixMode();
  if (mode != pto::FmatrixMode::FMATRIX_A_MANUAL &&
      mode != pto::FmatrixMode::FMATRIX_B_MANUAL) {
    return op.emitOpError("expects fmatrix_mode to be a_manual or b_manual");
  }
  return success();
}

LogicalResult mlir::pto::SetFmatrixOp::verify() {
  return verifySetFmatrixFamilyOp(*this);
}

LogicalResult mlir::pto::SetImg2colRptOp::verify() {
  return verifySetFmatrixFamilyOp(*this);
}

LogicalResult mlir::pto::SetImg2colPaddingOp::verify() {
  return verifySetFmatrixFamilyOp(*this);
}

// Both positions are independently optional. Two declarative optional groups
// starting with a comma cannot distinguish a K-only position from an M one.
static ParseResult parseImg2colPositions(
    OpAsmParser &parser, std::optional<OpAsmParser::UnresolvedOperand> &m,
    std::optional<OpAsmParser::UnresolvedOperand> &k) {
  while (succeeded(parser.parseOptionalComma())) {
    StringRef keyword;
    if (parser.parseKeyword(&keyword)) {
      return failure();
    }
    const bool isM = keyword == "pos_m";
    if (!isM && keyword != "pos_k") {
      return parser.emitError(parser.getCurrentLocation(), "expected 'pos_m' or 'pos_k'");
    }
    auto &position = isM ? m : k;
    if (position) {
      return parser.emitError(parser.getCurrentLocation()) << "duplicate " << keyword << " operand";
    }
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseEqual()) {
      return failure();
    }
    if (parser.parseOperand(operand)) {
      return failure();
    }
    position = operand;
  }
  return success();
}

static void printImg2colPositions(OpAsmPrinter &printer, Operation *, Value m, Value k) {
  if (m) {
    printer << ", pos_m = " << m;
  }
  if (k) {
    printer << ", pos_k = " << k;
  }
}

// Full NZ Mat bytes are [C/16, H*W, 16], exactly one NC1HWC0 image.
// Geometry belongs to the consuming op so the original Mat SSA retains its
// lifetime and synchronization identity in every memory planner.
static LogicalResult verifyTImg2colMatSource(TImg2colOp op) {
  if (!pto::isTargetArchA1(op.getOperation())) {
    return op.emitOpError("Mat image sources are supported only on A1");
  }
  auto source = cast<pto::TileBufType>(op.getSrc().getType());
  if (failed(verifyTileBufCommon(op, source, "src", false))) {
    return failure();
  }
  const auto shape = source.getShape();
  const auto space = getPTOMemorySpaceEnum(source);
  const bool alignedShape = shape.size() == 2 && llvm::all_of(shape, [](int64_t dim) {
    return dim >= 16 && dim <= 2048 && dim % 16 == 0;
  });
  const bool nzLayout =
      source.getBLayoutValueI32() == static_cast<int32_t>(pto::BLayout::ColMajor) &&
      source.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor) &&
      source.getSFractalSizeI32() == 512 && source.getCompactModeI32() == 0 && source.getPadValueI32() == 0;
  const bool completeImage = space && *space == pto::AddressSpace::MAT && source.getElementType().isF16() &&
                             alignedShape && source.getValidShape() == shape && nzLayout;
  if (!completeImage) {
    return op.emitOpError("Mat image requires a complete aligned FP16 NZ Mat tile");
  }
  const auto config = op.getConfig();
  const bool matchingGeometry = config && config->getFmapH().getInt() > 0 && config->getFmapW().getInt() > 0 &&
      config->getFmapH().getInt() * config->getFmapW().getInt() == shape[0] &&
      config->getChannelSize().getInt() == shape[1];
  if (!matchingGeometry) {
    return op.emitOpError("Mat image config must match [height * width, channels]");
  }
  return success();
}

static LogicalResult verifyTImg2colSrcConvLayout(TImg2colOp op) {
  auto srcCT = dyn_cast<pto::ConvTileBufType>(op.getSrc().getType());
  if (!srcCT) {
    return op.emitOpError("expects src to be a !pto.conv_tile");
  }
  auto srcLayout = dyn_cast_or_null<pto::ConvLayoutAttr>(srcCT.getLayout());
  if (!srcLayout ||
      (srcLayout.getValue() != pto::ConvLayout::NC1HWC0 &&
       srcLayout.getValue() != pto::ConvLayout::NDC1HWC0)) {
    return op.emitOpError("expects src layout to be NC1HWC0 or NDC1HWC0");
  }
  return success();
}

static LogicalResult verifyTImg2colDstLayout(TImg2colOp op) {
  auto dstTile = dyn_cast<pto::TileBufType>(op.getDst().getType());
  if (!dstTile) {
    return op.emitOpError("expects dst to be a !pto.tile_buf");
  }
  if (failed(verifyTileBufLayoutConstraints(op, dstTile, "dst"))) {
    return failure();
  }
  // The runtime generations disagree on the dst BLayout: A5's TIMG2COL
  // wants col_major (a5/TImg2col.hpp), A2/A3's wants row_major
  // (a2a3/TImg2col.hpp TIMG2COL_IMPL).
  const bool isA5 = getVerifierTargetArch(op.getOperation()) ==
                    VerifierTargetArch::A5;
  const auto expectedB = static_cast<int32_t>(isA5 ? pto::BLayout::ColMajor
                                                   : pto::BLayout::RowMajor);
  if (dstTile.getBLayoutValueI32() != expectedB ||
      dstTile.getSLayoutValueI32() !=
          static_cast<int32_t>(pto::SLayout::RowMajor)) {
    return op.emitOpError()
           << "expects dst layout to be BLayout="
           << (isA5 ? "col_major" : "row_major")
           << " and SLayout=row_major on this arch";
  }
  auto dstSpace = getPTOMemorySpaceEnum(dstTile);
  if (!dstSpace || *dstSpace != pto::AddressSpace::LEFT) {
    return op.emitOpError("expects dst to use loc=left");
  }
  return success();
}

static LogicalResult verifyTImg2colFmatrixMode(TImg2colOp op) {
  auto mode = op.getFmatrixMode();
  if (mode != pto::FmatrixMode::FMATRIX_A_AUTO &&
      mode != pto::FmatrixMode::FMATRIX_B_AUTO &&
      mode != pto::FmatrixMode::FMATRIX_A_MANUAL &&
      mode != pto::FmatrixMode::FMATRIX_B_MANUAL) {
    return op.emitOpError(
        "expects fmatrix_mode to be one of a_auto, b_auto, a_manual, or b_manual");
  }
  return success();
}

// Mirrors Timg2colConvTileCheck in a5/TImg2col.hpp: the tile dtypes are
// limited to int8/uint8/int16/uint16/int32/uint32/half/bfloat16/float. PTO
// integer types are signless, so the uint twins share one case each.
static LogicalResult verifyTImg2colElemType(TImg2colOp op) {
  // Mirrors Timg2colConvTileCheck: A5 accepts i8/i16/i32/f16/bf16/f32
  // (a5/TImg2col.hpp), while A2/A3's TIMG2COL_IMPL rejects i16/i32
  // (a2a3/TImg2col.hpp "Invalid data type").
  const bool isA5 = getVerifierTargetArch(op.getOperation()) ==
                    VerifierTargetArch::A5;
  Type elem = getElemTy(op.getDst().getType());
  const bool legal =
      elem.isInteger(8) || elem.isF16() || elem.isBF16() || elem.isF32() ||
      (isA5 && (elem.isInteger(16) || elem.isInteger(32)));
  if (!legal) {
    return op.emitOpError() << "expects timg2col element type to be "
                               "i8/f16/bf16/f32"
                            << (isA5 ? "/i16/i32" : "")
                            << " on this arch";
  }
  return success();
}

static LogicalResult verifyTImg2colFilterOrder(TImg2colOp op) {
  auto order = op.getFilterOrderAttr();
  if (!order) {
    return success();
  }
  const bool isA1 = pto::isTargetArchA1(op.getOperation());
  if (!isA1) {
    return op.emitOpError("filter_order is supported only on A1");
  }
  const auto taps = order.asArrayRef();
  const bool supportedCount = !taps.empty() && taps.size() <= 127;
  if (!supportedCount) {
    return op.emitOpError("filter_order requires 1..127 selected filter taps");
  }
  llvm::SmallSet<int64_t, 16> seen;
  for (int64_t tap : taps) {
    if (tap < 0 || tap >= 255 * 255) {
      return op.emitOpError("filter_order requires unique filter taps in [0, 65024]");
    }
    const bool inserted = seen.insert(tap).second;
    if (!inserted) {
      return op.emitOpError("filter_order requires unique filter taps in [0, 65024]");
    }
  }
  auto left = cast<pto::TileBufType>(op.getDst().getType());
  const int64_t columns = static_cast<int64_t>(taps.size()) * 16;
  if (left.getShape()[1] != columns) {
    return op.emitOpError("filter_order requires Left columns equal to 16 times its length");
  }
  return success();
}

LogicalResult mlir::pto::TImg2colOp::verify() {
  if (failed(verifyTileBufCommon(*this, getDst().getType(), "dst",
                                 /*allowLowPrecision=*/true))) {
    return failure();
  }
  if (failed(verifyTImg2colFilterOrder(*this))) {
    return failure();
  }
  const bool matSource = isa<pto::TileBufType>(getSrc().getType());
  if (matSource) {
    if (failed(verifyTImg2colMatSource(*this))) {
      return failure();
    }
  } else {
    if (getConfig()) {
      return emitOpError("ConvTile sources take config from their allocation");
    }
    if (failed(verifyConvTileCommon(*this, getSrc().getType(), "src", true))) {
      return failure();
    }
  }
  if (getElemTy(getDst().getType()) != getElemTy(getSrc().getType())) {
    return emitOpError() << "expects src and dst to have the same element type";
  }
  if (failed(verifyTImg2colElemType(*this)) ||
      (!matSource && failed(verifyTImg2colSrcConvLayout(*this))) ||
      failed(verifyTImg2colDstLayout(*this)) ||
      failed(verifyTImg2colFmatrixMode(*this))) {
    return failure();
  }

  // posM/posK are runtime offsets; only constant ones can be statically
  // range-checked against the uint16 the runtime takes.
  auto checkPos = [&](Value pos, StringRef name) -> LogicalResult {
    std::optional<int64_t> constant = mlir::getConstantIntValue(pos);
    if (constant && (*constant < 0 ||
                     *constant > std::numeric_limits<uint16_t>::max())) {
      return emitOpError() << "expects " << name
                           << " to fit in an unsigned 16-bit integer";
    }
    return success();
  };
  if (getPosM() && failed(checkPos(getPosM(), "posM"))) {
    return failure();
  }
  if (getPosK() && failed(checkPos(getPosK(), "posK"))) {
    return failure();
  }

  return success();
}
