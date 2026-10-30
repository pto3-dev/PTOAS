// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Included by PTO.cpp as part of the PTO IR implementation translation unit.

// A1 panel insertion uses MTE1 for Mat-to-Left, MTE3 for Vec-to-Mat,
// and V for Vec-to-Vec, distinct from FIXPIPE insertion.
static LogicalResult verifyTInsertA1(TInsertOp op) {
  auto common = verifyTInsertCommon(op, /*allowLowPrecision=*/false);
  if (failed(common)) return failure();
  const TInsertCommon &c = *common;
  const bool matToLeft = c.srcSpace && c.dstSpace && *c.srcSpace == pto::AddressSpace::MAT &&
                         *c.dstSpace == pto::AddressSpace::LEFT && c.srcElem.isF16() && c.dstElem.isF16();
  const bool vecToVec = c.srcSpace && c.dstSpace && *c.srcSpace == pto::AddressSpace::VEC &&
                        *c.dstSpace == pto::AddressSpace::VEC && c.srcElem == c.dstElem &&
                        (c.srcElem.isF16() || c.srcElem.isF32());
  const bool vecToMat = c.srcSpace && c.dstSpace && *c.srcSpace == pto::AddressSpace::VEC &&
                        *c.dstSpace == pto::AddressSpace::MAT && c.srcElem.isF16() && c.dstElem.isF16();
  if (!matToLeft && !vecToVec && !vecToMat)
    return op.emitOpError("A1 insertion requires FP16 Mat-to-Left, Vec-to-Mat, or same-type FP16/FP32 Vec operands");
  if (op.getFp() || op.getPreQuantScalar() || op.getAccToVecModeAttr() || op.getTinsertModeAttr() ||
      op.getReluPreMode() != pto::ReluPreMode::NoRelu)
    return op.emitOpError("A1 panel insertion does not support auxiliary operands or fused modes");
  return success();
}

::mlir::pto::PIPE mlir::pto::TInsertOp::getPipe() {
  auto isA5Target = [&]() -> bool {
    auto moduleOp = getOperation()->getParentOfType<::mlir::ModuleOp>();
    if (!moduleOp)
      return false;

    if (auto arch =
            moduleOp->getAttrOfType<::mlir::StringAttr>("pto.target_arch")) {
      if (arch.getValue().equals_insensitive("a5"))
        return true;
    }

    if (auto spec =
            moduleOp->getAttrOfType<::mlir::StringAttr>("pto.device-spec")) {
      auto s = spec.getValue();
      if (s.starts_with("Ascend950") || s.starts_with("Ascend910_95"))
        return true;
    }
    return false;
  };

  auto getASFromType = [](Type ty)
      -> std::optional<::mlir::pto::AddressSpace> {
    if (auto tb = ::mlir::dyn_cast<::mlir::pto::TileBufType>(ty)) {
      if (auto as = ::mlir::dyn_cast_or_null<::mlir::pto::AddressSpaceAttr>(
              tb.getMemorySpace()))
        return as.getAddressSpace();
      return std::nullopt;
    }
    return std::nullopt;
  };

  auto sOpt = getASFromType(getSrc().getType());
  auto dOpt = getASFromType(getDst().getType());
  if (::mlir::pto::isTargetArchA1(getOperation()) && sOpt && dOpt &&
      *sOpt == ::mlir::pto::AddressSpace::MAT && *dOpt == ::mlir::pto::AddressSpace::LEFT)
    return ::mlir::pto::PIPE::PIPE_MTE1;
  if (::mlir::pto::isTargetArchA1(getOperation()) && sOpt && dOpt &&
      *sOpt == ::mlir::pto::AddressSpace::VEC && *dOpt == ::mlir::pto::AddressSpace::MAT)
    return ::mlir::pto::PIPE::PIPE_MTE3;
  if (sOpt.has_value() && dOpt.has_value() &&
      sOpt.value() == ::mlir::pto::AddressSpace::VEC &&
      dOpt.value() == ::mlir::pto::AddressSpace::VEC)
    return ::mlir::pto::PIPE::PIPE_V;
  if (isA5Target() && sOpt.has_value() && dOpt.has_value() &&
      sOpt.value() == ::mlir::pto::AddressSpace::VEC &&
      dOpt.value() == ::mlir::pto::AddressSpace::MAT)
    return ::mlir::pto::PIPE::PIPE_MTE3;

  // Default and Acc->Mat path.
  return ::mlir::pto::PIPE::PIPE_FIX;

}

static LogicalResult verifyTInsertA5(TInsertOp op) {
  auto common = verifyTInsertCommon(op, /*allowLowPrecision=*/true);
  if (failed(common)) {
    return failure();
  }
  const TInsertCommon &c = *common;
  if (failed(verifyTInsertA5Modes(op, c))) {
    return failure();
  }
  if (*c.srcSpace == pto::AddressSpace::ACC &&
      (*c.dstSpace == pto::AddressSpace::MAT || *c.dstSpace == pto::AddressSpace::VEC)) {
    return verifyTInsertA5Acc(op, c);
  }
  if (*c.srcSpace == pto::AddressSpace::VEC && *c.dstSpace == pto::AddressSpace::MAT) {
    return verifyTInsertA5VecMat(op, c);
  }
  if (*c.srcSpace == pto::AddressSpace::VEC && *c.dstSpace == pto::AddressSpace::VEC) {
    return verifyTInsertA5VecVec(op, c);
  }
  return op.emitOpError(
      "expects A5 tinsert to use a supported src/dst loc pair: "
      "acc->mat, acc->vec, vec->mat, or vec->vec");
}

mlir::LogicalResult mlir::pto::TInsertOp::verify() {
  if (isTargetArchA1(getOperation())) return verifyTInsertA1(*this);
  auto verifyA2A3 = [this]() -> LogicalResult { return verifyTInsertA2A3(*this); };
  auto verifyA5 = [this]() -> LogicalResult { return verifyTInsertA5(*this); };
  return dispatchVerifierByArch(getOperation(), verifyA2A3, verifyA5);
}

static bool isColMajorRowMajorNZTileBuf(pto::TileBufType ty) {
  return ty.getBLayoutValueI32() != static_cast<int32_t>(pto::BLayout::RowMajor) &&
         ty.getSLayoutValueI32() == static_cast<int32_t>(pto::SLayout::RowMajor);
}

static bool isA5Fp8LikeType(Type ty) {
  if (auto ft = dyn_cast<FloatType>(ty)) {
      return ft.getWidth() == mlir::pto::kValue8;
  }
  return false;
}

static bool isA5MxFp8InputType(Type ty) {
  return ty && isa<Float8E4M3FNType, Float8E5M2Type>(ty);
}

static bool isA5MxInputTypePair(Type lhsTy, Type rhsTy) {
  return (isA5MxFp8InputType(lhsTy) && isA5MxFp8InputType(rhsTy)) ||
         (isPTOFloat4PackedType(lhsTy) && isPTOFloat4PackedType(rhsTy));
}

static LogicalResult verifyA5MxTypeTriple(Operation *op, Type lhsTy, Type rhsTy,
                                          Type dstTy, StringRef lhsName,
                                          StringRef rhsName, StringRef dstName) {
  Type lhsElem = getElemTy(lhsTy);
  Type rhsElem = getElemTy(rhsTy);
  Type dstElem = getElemTy(dstTy);

  if (!isA5MxInputTypePair(lhsElem, rhsElem)) {
    return op->emitOpError()
           << "expects A5 mx " << lhsName << "/" << rhsName
           << " element types to be a supported fp8/fp8 or fp4/fp4 pair";
  }

  if (!dstElem.isF32()) {
    return op->emitOpError()
           << "expects A5 mx result " << dstName << " to use f32 element type";
  }

  return success();
}

static bool isA5VectorPreQuantTypePair(Type srcElem, Type dstElem) {
  if (srcElem.isF32()) {
      return dstElem.isInteger(mlir::pto::kValue8) || isA5Fp8LikeType(dstElem) || isPTOHiFloat8Type(dstElem) ||
             dstElem.isF16() || dstElem.isBF16() || dstElem.isF32();
  }
  if (srcElem.isInteger(mlir::pto::kValue32)) {
      return dstElem.isInteger(mlir::pto::kValue8) || dstElem.isF16() || dstElem.isBF16();
  }
  return false;
}

static LogicalResult verifyTFillPadElementTypes(Operation *op, Type srcTy,
                                                Type dstTy) {
  auto srcElem = getElemTy(srcTy);
  auto dstElem = getElemTy(dstTy);
  int64_t srcB = getPTOStorageElemByteSize(srcElem);
  int64_t dstB = getPTOStorageElemByteSize(dstElem);
  if (srcB == 0 || dstB == 0) {
    return op->emitError("unsupported element type (expects int/float element types)");
  }
  if (srcB != dstB)
    return op->emitError("expects sizeof(src element) == sizeof(dst element)");
  if (!(srcB == 1 || srcB == mlir::pto::kValue2 || srcB == mlir::pto::kValue4))
    return op->emitError("expects element size to be 1, 2, or 4 bytes");
  return success();
}

static LogicalResult verifyTFillPadShapeExpansion(Operation *op, Type srcTy,
                                                  Type dstTy) {
  auto srcShape = getShapeVec(srcTy);
  auto dstShape = getShapeVec(dstTy);
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  bool expanded = false;
  for (auto [srcDim, dstDim] : llvm::zip_equal(srcShape, dstShape)) {
    if (srcDim == dstDim)
      continue;
    if (ShapedType::isDynamic(srcDim) || ShapedType::isDynamic(dstDim))
      return op->emitError("cannot infer TFILLPAD lowering from mismatched "
                           "dynamic physical shapes");
    if (srcDim > dstDim)
      return op->emitError(
          "expects each dst physical shape dimension to be >= src");
    expanded = true;
  }
  if (expanded &&
      (!srcSpace || !dstSpace || *srcSpace != pto::AddressSpace::VEC ||
       *dstSpace != pto::AddressSpace::VEC))
    return op->emitError("expects expanded TFILLPAD only for loc=vec");
  return success();
}

static LogicalResult verifyTFillPadMatTypes(Operation *op, Type srcTy,
                                            Type dstTy) {
  auto srcTb = mlir::dyn_cast<mlir::pto::TileBufType>(srcTy);
  auto dstTb = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy);
  auto srcSpace = getPTOMemorySpaceEnum(srcTy);
  auto dstSpace = getPTOMemorySpaceEnum(dstTy);
  if (srcTb && dstTb && srcSpace && dstSpace &&
      *srcSpace == mlir::pto::AddressSpace::MAT &&
      *dstSpace == mlir::pto::AddressSpace::MAT && srcTb != dstTb) {
    auto dimToStr = [](int64_t dim) -> std::string {
      return dim == ShapedType::kDynamic ? "?" : std::to_string(dim);
    };
    SmallVector<std::string, mlir::pto::kValue4> mismatchFields;
    auto srcValid = getValidShapeVec(srcTy);
    auto dstValid = getValidShapeVec(dstTy);
    if (srcValid.size() == mlir::pto::kValue2 &&
        dstValid.size() == mlir::pto::kValue2) {
      if (srcValid[0] != dstValid[0]) {
        mismatchFields.push_back("v_row (" + dimToStr(srcValid[0]) + " vs " +
                                 dimToStr(dstValid[0]) + ")");
      }
      if (srcValid[1] != dstValid[1]) {
        mismatchFields.push_back("v_col (" + dimToStr(srcValid[1]) + " vs " +
                                 dimToStr(dstValid[1]) + ")");
      }
    }
    if (srcTb.getPadValueI32() != dstTb.getPadValueI32()) {
      mismatchFields.push_back("pad (" + std::to_string(srcTb.getPadValueI32()) +
                               " vs " + std::to_string(dstTb.getPadValueI32()) +
                               ")");
    }

    auto diag = op->emitError()
                << "expects src/dst tile types to be lowerable to TFILLPAD "
                   "for loc=mat";
    if (!mismatchFields.empty()) {
      diag << "; mismatching fields: " << llvm::join(mismatchFields, ", ");
    }
    diag << "\n  src: " << srcTy;
    diag << "\n  dst: " << dstTy;
    diag << "\n  note: heterogeneous TFILLPAD overload is only available for loc=vec";
    return failure();
  }
  return success();
}

static mlir::LogicalResult verifyTFillPadLike(Operation *op, Type srcTy,
                                              Type dstTy) {
  if (!isPTOShapedLike(srcTy) || !isPTOShapedLike(dstTy))
    return op->emitError("expects src/dst to be PTO shaped-like types");
  if (getShapeVec(srcTy).size() != mlir::pto::kValue2 ||
      getShapeVec(dstTy).size() != mlir::pto::kValue2)
    return op->emitError("expects rank-2 shaped types for src/dst");
  if (failed(verifyTFillPadElementTypes(op, srcTy, dstTy)) ||
      failed(verifyTFillPadShapeExpansion(op, srcTy, dstTy)) ||
      failed(verifyTFillPadMatTypes(op, srcTy, dstTy)))
    return failure();
  if (auto dstTileTy = mlir::dyn_cast<mlir::pto::TileBufType>(dstTy)) {
    auto padAttr = mlir::dyn_cast<mlir::pto::PadValueAttr>(dstTileTy.getPadValueAttr());
    if (!padAttr || padAttr.getValue() == mlir::pto::PadValue::Null)
      return op->emitError("expects dst PadVal != Null for tfillpad");
  }
  return mlir::success();
}

mlir::LogicalResult mlir::pto::TFillPadOp::verify() {
  if (getOperation()->getAttr("mode")) {
    return emitOpError("does not accept 'mode'; PTOAS infers TFILLPAD lowering "
                       "from physical shape and planned addresses");
  }

  if (failed(verifyTFillPadLike(getOperation(), getSrc().getType(),
                                getDst().getType()))) {
    return failure();
  }

  if (auto padValueAttr = getPadValueAttr()) {
    auto dstSpace = getPTOMemorySpaceEnum(getDst().getType());
    if (!dstSpace || *dstSpace != pto::AddressSpace::MAT) {
      return emitOpError("expects padValue attribute only for loc=mat tfillpad");
    }
    auto dstTileTy = dyn_cast<pto::TileBufType>(getDst().getType());
    if (!dstTileTy) {
      return emitOpError("expects dst to be tile_buf when padValue is specified");
    }
    if (dstTileTy.getPadValueI32() != static_cast<int32_t>(padValueAttr.getValue())) {
      return emitOpError("expects padValue attribute to match dst tile pad configuration");
    }
  }

  return success();
}


static bool isSupportedGatherElemTypeA5Index(Type ty) {
  if (isPTOFloat8Type(ty)) {
    return true;
  }
  if (ty.isF16() || ty.isF32()) {
    return true;
  }
  if (auto it = dyn_cast<IntegerType>(ty)) {
    unsigned width = it.getWidth();
    return width == mlir::pto::kValue8 || width == mlir::pto::kValue16 ||
           width == mlir::pto::kValue32;
  }
  return false;
}

static unsigned getMaskGatherTimes(mlir::pto::MaskPatternAttr mp) {
  switch (mp.getValue()) {
  case mlir::pto::MaskPattern::P1111:
    return 1;
  case mlir::pto::MaskPattern::P0101:
  case mlir::pto::MaskPattern::P1010:
    return mlir::pto::kValue2;
  default:
    return mlir::pto::kValue4;
  }
}
