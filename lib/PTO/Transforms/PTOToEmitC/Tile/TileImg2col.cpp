// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "../PTOToEmitCEmitters.h"
#include "TileInternal.h"

#include <utility>

using namespace mlir;
using namespace mlir::pto;

#define DEBUG_TYPE "pto-emitc"

namespace mlir {
namespace pto {

struct PTOSetFmatrixToEmitC : public OpConversionPattern<pto::SetFmatrixOp> {
  using OpConversionPattern<pto::SetFmatrixOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SetFmatrixOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto templateArgs = getFmatrixModeTemplateArgs(
        rewriter, op.getFmatrixMode(),
        {*getEmitCConvTileBufTypeString(cast<pto::ConvTileBufType>(op.getSrc().getType()))});
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "SETFMATRIX",
                                 ValueRange{peelUnrealized(adaptor.getSrc())},
                                 ArrayAttr{}, templateArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSetImg2colRptToEmitC
    : public OpConversionPattern<pto::SetImg2colRptOp> {
  using OpConversionPattern<pto::SetImg2colRptOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SetImg2colRptOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto templateArgs = getFmatrixModeTemplateArgs(
        rewriter, op.getFmatrixMode(),
        {*getEmitCConvTileBufTypeString(cast<pto::ConvTileBufType>(op.getSrc().getType()))});
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "SET_IMG2COL_RPT",
                                 ValueRange{peelUnrealized(adaptor.getSrc())},
                                 ArrayAttr{}, templateArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOSetImg2colPaddingToEmitC
    : public OpConversionPattern<pto::SetImg2colPaddingOp> {
  using OpConversionPattern<pto::SetImg2colPaddingOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::SetImg2colPaddingOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto templateArgs = getFmatrixModeTemplateArgs(
        rewriter, op.getFmatrixMode(),
        {*getEmitCConvTileBufTypeString(cast<pto::ConvTileBufType>(op.getSrc().getType()))});
    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "SET_IMG2COL_PADDING",
                                 ValueRange{peelUnrealized(adaptor.getSrc())},
                                 ArrayAttr{}, templateArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

// Materialize only a descriptor: the planner and synchronization have already
// processed the real Mat source operand, and no storage or copy is introduced.
static std::pair<Value, std::string>
materializeImg2colSource(TImg2colOp op, Value source,
                        ConversionPatternRewriter &rewriter) {
  if (auto conv = dyn_cast<ConvTileBufType>(op.getSrc().getType())) {
    return {source, *getEmitCConvTileBufTypeString(conv)};
  }
  auto mat = cast<TileBufType>(op.getSrc().getType());
  const auto config = *op.getConfig();
  const auto shape = mat.getShape();
  auto *ctx = rewriter.getContext();
  SmallVector<int64_t, 5> imageShape{1, shape[1] / 16, config.getFmapH().getInt(),
                                    config.getFmapW().getInt(), 16};
  auto conv = ConvTileBufType::get(ctx, imageShape, mat.getElementType(),
      rewriter.getI64IntegerAttr(shape[0] * shape[1]),
      AddressSpaceAttr::get(ctx, AddressSpace::MAT),
      ConvLayoutAttr::get(ctx, ConvLayout::NC1HWC0));
  const auto typeName = *getEmitCConvTileBufTypeString(conv);
  auto type = emitc::OpaqueType::get(ctx, typeName);
  Value descriptor = rewriter.create<emitc::VariableOp>(
      op.getLoc(), getEmitCVariableResultType(type), emitc::OpaqueAttr::get(ctx, "")).getResult();
  descriptor = loadEmitCVariableIfNeeded(rewriter, op.getLoc(), descriptor);
  emitConvTileConfigInitCall(rewriter, op.getLoc(), descriptor, config);
  Value pointer = materializeTileDataValue(rewriter, op.getLoc(), source, AddressSpace::MAT, "half");
  assignTileAddress(rewriter, op.getLoc(), ctx, descriptor, pointer);
  return {descriptor, typeName};
}

struct PTOTImg2colToEmitC : public OpConversionPattern<pto::TImg2colOp> {
  using OpConversionPattern<pto::TImg2colOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pto::TImg2colOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    Value dst = peelUnrealized(adaptor.getDst());
    auto source = materializeImg2colSource(op, peelUnrealized(adaptor.getSrc()), rewriter);
    Value src = source.first;
    Type u16Ty = emitc::OpaqueType::get(ctx, "uint16_t");
    // posM/posK are runtime offsets. A constant operand folds to the same
    // uint16_t constant the old attribute form emitted; absent means 0.
    // Either way a non-constant operand passes through as a cast so a loop
    // induction variable can drive the in-loop Split-K pattern.
    auto materializePos = [&](Value operand) -> Value {
      if (!operand) {
        return makeEmitCIntConstant(rewriter, loc, u16Ty, 0);
      }
      operand = peelUnrealized(operand);
      return rewriter
          .create<emitc::CallOpaqueOp>(
              loc, u16Ty, "static_cast", ArrayAttr{},
              rewriter.getArrayAttr(
                  {emitc::OpaqueAttr::get(ctx, "uint16_t")}),
              ValueRange{operand})
          .getResult(0);
    };
    Value posM = materializePos(adaptor.getPosM());
    Value posK = materializePos(adaptor.getPosK());
    auto templateArgs = getFmatrixModeTemplateArgs(
        rewriter, op.getFmatrixMode(),
        {*getEmitCTileTypeString(cast<pto::TileBufType>(op.getDst().getType())),
         source.second});

    createLastUseAwareOpaqueCall(rewriter, op.getOperation(), TypeRange{},
                                 "TIMG2COL",
                                 ValueRange{dst, src, posM, posK}, ArrayAttr{},
                                 templateArgs);
    rewriter.eraseOp(op);
    return success();
  }
};

void populateTileTileImg2colPatterns(RewritePatternSet &patterns,
                        TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.add<PTOSetFmatrixToEmitC, PTOSetImg2colRptToEmitC,
               PTOSetImg2colPaddingToEmitC, PTOTImg2colToEmitC>(
      typeConverter, ctx);
}

} // namespace pto
} // namespace mlir
