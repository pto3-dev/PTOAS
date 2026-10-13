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
    if (!rows || !cols || *rows <= 0 || *cols <= 0 || rowStride != cols || colStride != 1 ||
        *cols % (32 / vectorElementBytes(type.getElementType())))
        return view.emitError("A1 vector lowering requires static contiguous aligned tensor views.");
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
    auto offset = partition.getOffsets().size() == 2 ? constantInt(partition.getOffsets()[1]) : std::nullopt;
    auto view = partition.getSource().getDefiningOp<MakeTensorViewOp>();
    if (!rows || !cols || *rows != tile.getShape()[0] || *cols != tile.getShape()[1] || !offset ||
        *offset % (32 / vectorElementBytes(tile.getElementType())) || !view ||
        constantInt(view.getStrides()[0]) != cols)
        return op->emitError("A1 DMA requires matching static tile sizes, contiguous rows and aligned column offsets.");
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
                SetFlagOp, WaitFlagOp, BarrierOp, FenceBarrierAllOp, GetBlockIdxOp, GetBlockNumOp>(op))
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
            if (failed(validateTransfer(op, load.getSrc(), load.getDst())))
                return WalkResult::interrupt();
        }
        if (auto store = dyn_cast<TStoreOp>(op)) {
            if (store.getFp() || store.getPreQuantScalar() || store.getAtomicType() != AtomicType::AtomicNone ||
                store.getStPhase() != STPhase::Unspecified || store.getReluPreMode() != ReluPreMode::NoRelu)
                return reject("A1 store quantization, atomics, unit flags and relu modes are unavailable.");
            if (failed(validateTransfer(op, store.getDst(), store.getSrc())))
                return WalkResult::interrupt();
        }
        if (isa<TAddOp, TSubOp, TMulOp>(op)) {
            for (Type type : op->getOperandTypes())
                if (auto tile = dyn_cast<TileBufType>(type))
                    if ((tile.getShape()[0] * tile.getShape()[1]) % (256 / vectorElementBytes(tile.getElementType())))
                        return reject("A1 binary vector arithmetic requires complete 256-byte repeats.");
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
