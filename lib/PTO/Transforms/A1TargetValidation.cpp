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
        if (shape.size() != 2 || shape[0] != 16 || shape[1] != 16 || tile.getValidShape() != shape ||
            tile.getCompactModeI32() != 0)
            return op->emitError("A1 matrix lowering requires static full single-fractal 16x16 tiles.");
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
            supported = tile.getElementType().isF16() && tile.getBLayoutValueI32() == row &&
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
                const uint64_t bytes = 256 * vectorElementBytes(tile.getElementType());
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
    if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0 || tile.getValidShape() != shape ||
        tile.getBLayoutValueI32() != 0 || tile.getSLayoutValueI32() != 0 || tile.getCompactModeI32() != 0)
        return op->emitError("A1 vector lowering requires static full row-major tile shapes.");
    const unsigned elementBytes = vectorElementBytes(tile.getElementType());
    if (shape[1] % (32 / elementBytes) || shape[0] > static_cast<int64_t>(kA1UBCapacityBytes / elementBytes) / shape[1])
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
    if ((view.getLayoutAttr() && view.getLayoutAttr().getLayout() != Layout::ND) ||
        (type.getLayoutAttr() && type.getLayoutAttr().getLayout() != Layout::ND))
        return view.emitError("A1 vector DMA requires ND tensor views.");
    if (view.getShape().size() != 2 || view.getStrides().size() != 2)
        return view.emitError("A1 vector lowering requires two-dimensional tensor views.");
    auto rows = constantInt(view.getShape()[0]);
    auto cols = constantInt(view.getShape()[1]);
    auto rowStride = constantInt(view.getStrides()[0]);
    auto colStride = constantInt(view.getStrides()[1]);
    if (!rows || !cols || !rowStride || *rows <= 0 || *cols <= 0 || *rowStride < *cols || colStride != 1 ||
        *cols % (32 / vectorElementBytes(type.getElementType())) ||
        *rowStride % (32 / vectorElementBytes(type.getElementType())))
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
                TMovOp, TMatmulOp, SectionCubeOp, SectionVectorOp, SetFlagOp, WaitFlagOp, BarrierOp, FenceBarrierAllOp,
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
            if (location == AddressSpace::MAT) {
                auto partition = load.getSrc().getDefiningOp<PartitionViewOp>();
                auto view = partition ? partition.getSource().getDefiningOp<MakeTensorViewOp>() : MakeTensorViewOp{};
                if (!view || view.getStrides().size() != 2 || constantInt(view.getStrides()[0]) != 16)
                    return reject("A1 matrix loads require contiguous 16-element tensor rows.");
            }
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
        if (isa<TAddOp, TSubOp, TMulOp>(op))
            for (Value operand : op->getOperands())
                if (locationOf(operand) != AddressSpace::VEC)
                    return reject("A1 vector arithmetic requires Vec operands.");
        if (auto move = dyn_cast<TMovOp>(op)) {
            const auto src = locationOf(move.getSrc());
            const auto dst = locationOf(move.getDst());
            if (move.getFp() || move.getPreQuantScalar() || move.getGrpAxisAttr() || move.getNumResults() ||
                move.getReluPreMode() != ReluPreMode::NoRelu)
                return reject("A1 matrix moves do not support auxiliary operands, results or fused modes.");
            if (src == AddressSpace::MAT && (dst == AddressSpace::LEFT || dst == AddressSpace::RIGHT)) {
                if (move.getAccToVecModeAttr())
                    return reject("A1 operand moves do not accept accumulator modes.");
            } else if (src == AddressSpace::ACC && dst == AddressSpace::VEC) {
                auto tile = cast<TileBufType>(move.getDst().getType());
                if (tile.getShape()[0] != 16 || tile.getShape()[1] != 16 || !tile.getElementType().isF32() ||
                    !move.getAccToVecModeAttr() ||
                    move.getAccToVecModeAttr().getValue() != AccToVecMode::SingleModeVec0)
                    return reject("A1 writeback requires FP32 ND 16x16 Vec and explicit single_mode_vec0.");
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
