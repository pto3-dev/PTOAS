// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/A1Target.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Matchers.h"

using namespace mlir;
using namespace mlir::pto;

namespace {
std::optional<int64_t> constantInt(Value value)
{
    if (!value)
        return std::nullopt;
    APInt integer;
    if (!matchPattern(value, m_ConstantInt(&integer)))
        return std::nullopt;
    return integer.getSExtValue();
}

unsigned vectorElementBytes(Type element) { return element.isF16() ? 2 : 4; }

LogicalResult validateTile(Operation* op, TileBufType tile)
{
    auto shape = tile.getShape();
    auto space = dyn_cast_or_null<AddressSpaceAttr>(tile.getMemorySpace());
    if (space && space.getAddressSpace() != AddressSpace::VEC) {
        const auto location = space.getAddressSpace();
        if (shape.size() != 2 || shape[0] < 16 || shape[0] > 2048 || shape[0] % 16 || shape[1] != 16 ||
            tile.getValidShape() != shape || tile.getCompactModeI32() != 0)
            return op->emitError("A1 narrow matrix lowering requires full aligned rows in [16, 2048] and 16 columns.");
        const auto row = static_cast<int32_t>(BLayout::RowMajor);
        const auto col = static_cast<int32_t>(BLayout::ColMajor);
        const auto innerRow = static_cast<int32_t>(SLayout::RowMajor);
        const auto innerCol = static_cast<int32_t>(SLayout::ColMajor);
        bool supported = false;
        if (location == AddressSpace::MAT)
            supported = tile.getElementType().isF16() &&
                        (tile.getBLayoutValueI32() == row || tile.getBLayoutValueI32() == col) &&
                        tile.getSLayoutValueI32() == innerRow && tile.getSFractalSizeI32() == 512;
        if (location == AddressSpace::LEFT)
            supported = tile.getElementType().isF16() && tile.getBLayoutValueI32() == col &&
                        tile.getSLayoutValueI32() == innerRow && tile.getSFractalSizeI32() == 512;
        if (location == AddressSpace::RIGHT)
            supported = shape[0] == 16 && tile.getElementType().isF16() && tile.getBLayoutValueI32() == row &&
                        tile.getSLayoutValueI32() == innerCol && tile.getSFractalSizeI32() == 512;
        if (location == AddressSpace::ACC)
            supported = tile.getElementType().isF32() && tile.getBLayoutValueI32() == col &&
                        tile.getSLayoutValueI32() == innerRow && tile.getSFractalSizeI32() == 1024;
        if (!supported)
            return op->emitError("A1 matrix tiles require FP16 ZZ/NZ Mat, NZ Left/ZN Right or FP32 NZ Acc layouts.");
        if (auto alloc = dyn_cast<AllocTileOp>(op)) {
            if (alloc.getAddr()) {
                auto addr = constantInt(alloc.getAddr());
                const auto spec = getA1MemorySpec(location);
                const uint64_t bytes = shape[0] * 16 * vectorElementBytes(tile.getElementType());
                if (!addr || *addr < 0 || *addr % spec.alignmentBytes ||
                    static_cast<uint64_t>(*addr) > spec.capacityBytes - bytes)
                    return op->emitError(
                        "A1 matrix tile address must be static, aligned and within its memory budget.");
            }
        }
        return success();
    }
    if (!space || space.getAddressSpace() != AddressSpace::VEC ||
        (!tile.getElementType().isF16() && !tile.getElementType().isF32()))
        return op->emitError("A1 vector lowering requires FP16 or FP32 Vec tiles.");
    const bool packedColumn = shape.size() == 2 && shape[1] == 1 && tile.getBLayoutValueI32() == 1;
    if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0 || tile.getValidShape() != shape ||
        (!packedColumn && tile.getBLayoutValueI32() != 0) || tile.getSLayoutValueI32() != 0 ||
        tile.getCompactModeI32() != 0)
        return op->emitError("A1 vector lowering requires static full row-major tile shapes.");
    const unsigned elementBytes = vectorElementBytes(tile.getElementType());
    if ((packedColumn ? shape[0] : shape[1]) % (32 / elementBytes) ||
        shape[0] > static_cast<int64_t>(kA1UBCapacityBytes / elementBytes) / shape[1])
        return op->emitError("A1 vector tile must have 32-byte-aligned rows and fit the 128 KiB UB budget.");
    if (auto alloc = dyn_cast<AllocTileOp>(op)) {
        if (alloc.getAddr()) {
            auto addr = constantInt(alloc.getAddr());
            uint64_t bytes = static_cast<uint64_t>(shape[0] * shape[1]) * elementBytes;
            if (!addr || *addr < 0 || *addr % 32 || static_cast<uint64_t>(*addr) > kA1UBCapacityBytes - bytes)
                return op->emitError("A1 tile address must be static, aligned, and within the 128 KiB UB budget.");
        }
    }
    return success();
}

LogicalResult validateView(MakeTensorViewOp view)
{
    auto type = cast<TensorViewType>(view.getResult().getType());
    if (view.getShape().size() != 2 || view.getStrides().size() != 2)
        return view.emitError("A1 vector lowering requires two-dimensional tensor views.");
    auto rows = constantInt(view.getShape()[0]);
    auto cols = constantInt(view.getShape()[1]);
    auto rowStride = constantInt(view.getStrides()[0]);
    auto colStride = constantInt(view.getStrides()[1]);
    const unsigned block = 32 / vectorElementBytes(type.getElementType());
    const bool packedColumn =
        rows && cols == 1 && rowStride == 1 && colStride && *colStride > 0 && *rows > 0 && *rows % block == 0;
    auto supportedLayout = [&](Layout layout) {
        return layout == Layout::ND || (packedColumn && layout == Layout::DN);
    };
    if ((view.getLayoutAttr() && !supportedLayout(view.getLayoutAttr().getLayout())) ||
        (type.getLayoutAttr() && !supportedLayout(type.getLayoutAttr().getLayout())))
        return view.emitError("A1 vector DMA requires ND tensor views or contiguous packed DN columns.");
    if (!packedColumn && (!rows || !cols || !rowStride || *rows <= 0 || *cols <= 0 || *rowStride < *cols ||
                          colStride != 1 || *cols % block || *rowStride % block))
        return view.emitError("A1 vector lowering requires static non-overlapping 32-byte-aligned tensor rows.");
    return success();
}

LogicalResult validateTransfer(Operation* op, Value global, Value buffer)
{
    auto tile = dyn_cast<TileBufType>(buffer.getType());
    auto partition = global.getDefiningOp<PartitionViewOp>();
    if (!tile || !partition || partition.getSizes().size() != 2)
        return op->emitError("A1 DMA requires a two-dimensional partition matching its tile.");
    auto rows = constantInt(partition.getSizes()[0]);
    auto cols = constantInt(partition.getSizes()[1]);
    if (partition.getOffsets().size() != 2)
        return op->emitError("A1 DMA requires two partition offsets.");
    const int64_t rowOffset = constantInt(partition.getOffsets()[0]).value_or(-1);
    const int64_t colOffset = constantInt(partition.getOffsets()[1]).value_or(-1);
    auto view = partition.getSource().getDefiningOp<MakeTensorViewOp>();
    if (!rows || !cols || *rows != tile.getShape()[0] || *cols != tile.getShape()[1] || !view || rowOffset < 0 ||
        colOffset < 0 || colOffset % (32 / vectorElementBytes(tile.getElementType())))
        return op->emitError("A1 DMA requires matching static tile sizes and nonnegative aligned partition offsets.");
    if (failed(validateView(view)))
        return failure();
    const bool packedTile = tile.getShape()[1] == 1 && tile.getBLayoutValueI32() == 1;
    if (packedTile && (constantInt(view.getShape()[1]) != 1 || constantInt(view.getStrides()[0]) != 1 ||
                       rowOffset % (32 / vectorElementBytes(tile.getElementType()))))
        return op->emitError("A1 packed column DMA requires a contiguous Mx1 view and aligned row offset.");
    const int64_t viewRows = *constantInt(view.getShape()[0]);
    const int64_t viewCols = *constantInt(view.getShape()[1]);
    if (*rows > viewRows || *cols > viewCols || rowOffset > viewRows - *rows || colOffset > viewCols - *cols)
        return op->emitError("A1 DMA partition must fit within its source tensor view.");
    return success();
}

LogicalResult validateType(Operation* op, Type type)
{
    if (auto tile = dyn_cast<TileBufType>(type))
        return validateTile(op, tile);
    Type element;
    if (auto ptr = dyn_cast<PtrType>(type))
        element = ptr.getElementType();
    if (auto tensor = dyn_cast<TensorViewType>(type))
        element = tensor.getElementType();
    if (auto partition = dyn_cast<PartitionTensorViewType>(type))
        element = partition.getElementType();
    if (element && !element.isF16() && !element.isF32())
        return op->emitError("A1 vector lowering requires FP16 or FP32 tensor data.");
    return success();
}
} // namespace

void mlir::pto::normalizeA1StaticValidShapes(ModuleOp module)
{
    if (!isTargetArchA1(module))
        return;
    module.walk([](AllocTileOp alloc) {
        auto type = dyn_cast<TileBufType>(alloc.getResult().getType());
        if (!type || type.getShape().size() != 2 || type.getValidShape() == type.getShape())
            return;
        auto shape = type.getShape();
        auto valid = type.getValidShape();
        if (valid.size() != 2 || shape[0] <= 0 || shape[1] <= 0)
            return;
        auto row =
            ShapedType::isDynamic(valid[0]) ? constantInt(alloc.getValidRow()) : std::optional<int64_t>(valid[0]);
        auto col =
            ShapedType::isDynamic(valid[1]) ? constantInt(alloc.getValidCol()) : std::optional<int64_t>(valid[1]);
        if (row != shape[0] || col != shape[1])
            return;
        // Each SSA use observes the refined allocation type. Explicit valid
        // operands are redundant once the same extents live in the type.
        alloc.getResult().setType(TileBufType::get(
            type.getContext(), shape, type.getElementType(), type.getMemorySpace(), shape, type.getConfigAttr()));
        alloc.getValidRowMutable().clear();
        alloc.getValidColMutable().clear();
    });
}

LogicalResult mlir::pto::validateA1Capabilities(ModuleOp module)
{
    if (!isTargetArchA1(module))
        return success();
    WalkResult result = module.walk([&](Operation* op) -> WalkResult {
        auto reject = [&](StringRef message) {
            op->emitError(message);
            return WalkResult::interrupt();
        };
        if (auto child = dyn_cast<ModuleOp>(op)) {
            if (child != module) {
                if (auto arch = child->getAttrOfType<StringAttr>("pto.target_arch"))
                    if (!arch.getValue().equals_insensitive("a1"))
                        return reject("A1 lowering does not support mixed target architectures.");
            }
        }
        if (op->getName().getDialectNamespace() == "pto" &&
            !isa<
                AddPtrOp, MakeTensorViewOp, PartitionViewOp, AllocTileOp, TLoadOp, TStoreOp, TAddOp, TSubOp, TMulOp,
                TDivOp, TExpOp, TExpandsOp, TRowSumOp, TRowMaxOp, TColSumOp, TColMaxOp, TMovOp, TMatmulOp, TMatmulAccOp,
                TCvtOp, SectionCubeOp, SectionVectorOp, SetFlagOp, WaitFlagOp, BarrierOp, FenceBarrierAllOp,
                GetBlockIdxOp, GetBlockNumOp>(op))
            return reject("operation is not yet supported by the A1 EmitC profile.");
        for (Type type : op->getOperandTypes())
            if (failed(validateType(op, type)))
                return WalkResult::interrupt();
        for (Type type : op->getResultTypes())
            if (failed(validateType(op, type)))
                return WalkResult::interrupt();
        for (Region& region : op->getRegions())
            for (Block& block : region)
                for (BlockArgument arg : block.getArguments())
                    if (failed(validateType(op, arg.getType())))
                        return WalkResult::interrupt();
        if (auto view = dyn_cast<MakeTensorViewOp>(op))
            if (failed(validateView(view)))
                return WalkResult::interrupt();
        if (auto load = dyn_cast<TLoadOp>(op)) {
            if (load.getPadModeAttr() || load.getPadValue() || load.getLeftPaddingNum() || load.getRightPaddingNum() ||
                load.getInitOutBuffer() || load.getInitCondition() || load.getCachePolicyAttr())
                return reject("A1 load padding, initialization and cache-policy modes are unavailable.");
            const auto tile = dyn_cast<TileBufType>(load.getDst().getType());
            if (!tile)
                return reject("A1 loads require tile-buffer destinations.");
            const auto location = cast<AddressSpaceAttr>(tile.getMemorySpace()).getAddressSpace();
            if (location != AddressSpace::VEC && location != AddressSpace::MAT)
                return reject("A1 loads require a Vec or Mat destination.");
            if (failed(validateTransfer(op, load.getSrc(), load.getDst())))
                return WalkResult::interrupt();
        }
        if (auto store = dyn_cast<TStoreOp>(op)) {
            if (store.getFp() || store.getPreQuantScalar() || store.getAtomicType() != AtomicType::AtomicNone ||
                store.getStPhase() != STPhase::Unspecified || store.getReluPreMode() != ReluPreMode::NoRelu)
                return reject("A1 store quantization, atomics, unit flags and relu modes are unavailable.");
            const auto tile = dyn_cast<TileBufType>(store.getSrc().getType());
            if (!tile)
                return reject("A1 stores require tile-buffer sources.");
            if (cast<AddressSpaceAttr>(tile.getMemorySpace()).getAddressSpace() != AddressSpace::VEC)
                return reject("A1 stores require explicit accumulator-to-vector writeback.");
            if (failed(validateTransfer(op, store.getDst(), store.getSrc())))
                return WalkResult::interrupt();
        }
        auto locationOf = [](Value value) -> std::optional<AddressSpace> {
            auto tile = dyn_cast<TileBufType>(value.getType());
            auto space = tile ? dyn_cast_or_null<AddressSpaceAttr>(tile.getMemorySpace()) : AddressSpaceAttr{};
            return space ? std::optional<AddressSpace>(space.getAddressSpace()) : std::nullopt;
        };
        if (isa<TAddOp, TSubOp, TMulOp, TDivOp, TExpOp>(op))
            for (Value operand : op->getOperands())
                if (locationOf(operand) != AddressSpace::VEC ||
                    cast<TileBufType>(operand.getType()).getBLayoutValueI32() != 0)
                    return reject("A1 vector arithmetic requires row-major Vec operands.");
        if (isa<TRowSumOp, TRowMaxOp, TColSumOp, TColMaxOp>(op)) {
            for (Value operand : op->getOperands())
                if (locationOf(operand) != AddressSpace::VEC)
                    return reject("A1 reductions require Vec operands.");
            for (unsigned i = 0; i < op->getNumOperands(); ++i) {
                for (unsigned j = i + 1; j < op->getNumOperands(); ++j) {
                    Value a = op->getOperand(i), b = op->getOperand(j);
                    auto aa = a.getDefiningOp<AllocTileOp>(), ba = b.getDefiningOp<AllocTileOp>();
                    const int64_t ap = aa ? constantInt(aa.getAddr()).value_or(-1) : -1;
                    const int64_t bp = ba ? constantInt(ba.getAddr()).value_or(-1) : -1;
                    auto at = cast<TileBufType>(a.getType()), bt = cast<TileBufType>(b.getType());
                    const int64_t ab = at.getShape()[0] * at.getShape()[1] * vectorElementBytes(at.getElementType());
                    const int64_t bb = bt.getShape()[0] * bt.getShape()[1] * vectorElementBytes(bt.getElementType());
                    const bool firstRowAlias = isa<TColSumOp, TColMaxOp>(op) && op->getNumOperands() == 2 &&
                                               bt.getShape()[0] == 1 && at.getShape()[1] == bt.getShape()[1] &&
                                               (a == b || (ap >= 0 && bp == ap));
                    if (!firstRowAlias && (a == b || (ap >= 0 && bp >= 0 && ap < bp + bb && bp < ap + ab)))
                        return reject("A1 reductions require disjoint storage except exact first-row column results.");
                }
            }
            if (isa<TRowSumOp, TRowMaxOp>(op)) {
                if (op->getNumOperands() != 3)
                    return reject("A1 row reductions require explicit full-size scratch.");
                auto source = cast<TileBufType>(op->getOperand(0).getType());
                auto scratch = cast<TileBufType>(op->getOperand(1).getType());
                if (source.getBLayoutValueI32() != 0 || scratch.getBLayoutValueI32() != 0 ||
                    scratch.getShape()[0] < source.getShape()[0] || scratch.getShape()[1] < source.getShape()[1])
                    return reject("A1 row reductions require row-major source and full-size row-major scratch.");
            }
        }
        if (auto sum = dyn_cast<TColSumOp>(op))
            if (sum.getTmp() || sum.getIsBinary())
                return reject("A1 column sum supports the sequential form without scratch only.");
        if (auto fill = dyn_cast<TExpandsOp>(op))
            if (locationOf(fill.getDst()) != AddressSpace::VEC ||
                cast<TileBufType>(fill.getDst().getType()).getBLayoutValueI32() != 0)
                return reject("A1 scalar broadcast requires a Vec destination.");
        if (auto divide = dyn_cast<TDivOp>(op))
            if (divide.getPrecisionType() != DivPrecision::Default)
                return reject("A1 division supports default precision only.");
        if (auto exponential = dyn_cast<TExpOp>(op))
            if (exponential.getPrecisionType() != ExpPrecision::Default)
                return reject("A1 exponential supports default precision only.");
        if (auto move = dyn_cast<TMovOp>(op)) {
            const auto src = locationOf(move.getSrc());
            const auto dst = locationOf(move.getDst());
            if (move.getFp() || move.getPreQuantScalar() || move.getGrpAxisAttr() || move.getNumResults() ||
                move.getReluPreMode() != ReluPreMode::NoRelu)
                return reject("A1 matrix moves do not support auxiliary operands, results or fused modes.");
            if (src == AddressSpace::MAT && (dst == AddressSpace::LEFT || dst == AddressSpace::RIGHT)) {
                if (cast<TileBufType>(move.getSrc().getType()).getShape() !=
                    cast<TileBufType>(move.getDst().getType()).getShape())
                    return reject("A1 matrix operand moves require matching full shapes.");
                if (move.getAccToVecModeAttr())
                    return reject("A1 operand moves do not accept accumulator modes.");
            } else if (src == AddressSpace::ACC && dst == AddressSpace::VEC) {
                auto tile = cast<TileBufType>(move.getDst().getType());
                if (tile.getShape()[0] != cast<TileBufType>(move.getSrc().getType()).getShape()[0] ||
                    tile.getShape()[1] != 16 || !tile.getElementType().isF32() || !move.getAccToVecModeAttr() ||
                    move.getAccToVecModeAttr().getValue() != AccToVecMode::SingleModeVec0)
                    return reject("A1 writeback requires matching FP32 ND Nx16 Vec and explicit single_mode_vec0.");
            } else {
                return reject("A1 matrix moves require Mat-to-Left/Right or Acc-to-Vec.");
            }
        }
        if (auto matmul = dyn_cast<TMatmulOp>(op)) {
            if (locationOf(matmul.getLhs()) != AddressSpace::LEFT ||
                locationOf(matmul.getRhs()) != AddressSpace::RIGHT ||
                locationOf(matmul.getDst()) != AddressSpace::ACC || matmul.getNumResults() ||
                matmul.getAccPhase() != AccPhase::Unspecified)
                return reject("A1 matrix multiplication requires Left/Right-to-Acc without unit flags or results.");
        }
        if (auto matmul = dyn_cast<TMatmulAccOp>(op)) {
            if (locationOf(matmul.getAccIn()) != AddressSpace::ACC ||
                locationOf(matmul.getLhs()) != AddressSpace::LEFT ||
                locationOf(matmul.getRhs()) != AddressSpace::RIGHT ||
                locationOf(matmul.getDst()) != AddressSpace::ACC || matmul.getNumResults() ||
                matmul.getAccPhase() != AccPhase::Unspecified)
                return reject("A1 matrix accumulation requires Acc/Left/Right without unit flags or results.");
            if (matmul.getAccIn() != matmul.getDst())
                return reject("A1 matrix accumulation requires the same input/output accumulator SSA buffer.");
        }
        if (auto convert = dyn_cast<TCvtOp>(op)) {
            auto source = dyn_cast<TileBufType>(convert.getSrc().getType());
            auto destination = dyn_cast<TileBufType>(convert.getDst().getType());
            if (!source || !destination)
                return reject("A1 conversion requires tile-buffer operands.");
            if (locationOf(convert.getSrc()) != AddressSpace::VEC ||
                locationOf(convert.getDst()) != AddressSpace::VEC ||
                source.getElementType() == destination.getElementType() || source.getBLayoutValueI32() != 0 ||
                destination.getBLayoutValueI32() != 0 || convert.getTmp() || convert.getRmode() != RoundMode::NONE ||
                convert.getSatMode() != SaturationMode::OFF)
                return reject(
                    "A1 conversion requires an FP16/FP32 Vec pair, NONE rounding, saturation OFF and no scratch.");
            auto sourceAlloc = convert.getSrc().getDefiningOp<AllocTileOp>();
            auto destinationAlloc = convert.getDst().getDefiningOp<AllocTileOp>();
            auto sourceAddress = sourceAlloc ? constantInt(sourceAlloc.getAddr()) : std::nullopt;
            auto destinationAddress = destinationAlloc ? constantInt(destinationAlloc.getAddr()) : std::nullopt;
            if (sourceAddress && destinationAddress) {
                const auto sourceBytes =
                    source.getShape()[0] * source.getShape()[1] * vectorElementBytes(source.getElementType());
                const auto destinationBytes = destination.getShape()[0] * destination.getShape()[1] *
                                              vectorElementBytes(destination.getElementType());
                const bool disjoint = *sourceAddress + sourceBytes <= *destinationAddress ||
                                      *destinationAddress + destinationBytes <= *sourceAddress;
                const bool narrowingAlias =
                    destination.getElementType().isF16() && *sourceAddress == *destinationAddress;
                if (!disjoint && !narrowingAlias)
                    return reject("A1 conversion requires disjoint buffers or exact-alias FP32-to-FP16 narrowing.");
            }
        }
        if (auto event = dyn_cast_or_null<EventAttr>(op->getAttr("event_id")))
            if (static_cast<unsigned>(event.getEvent()) == kA1ReservedEventId)
                return reject("A1 event ID 7 is reserved for ISA helpers.");
        for (NamedAttribute attribute : op->getAttrs())
            if (auto pipe = dyn_cast<PipeAttr>(attribute.getValue()))
                if (static_cast<unsigned>(pipe.getPipe()) > static_cast<unsigned>(PIPE::PIPE_ALL) ||
                    (isa<SetFlagOp, WaitFlagOp>(op) && pipe.getPipe() == PIPE::PIPE_ALL))
                    return reject("A1 requires local supported pipelines; FIX and cross-core paths are unavailable.");
        return WalkResult::advance();
    });
    return result.wasInterrupted() ? failure() : success();
}
