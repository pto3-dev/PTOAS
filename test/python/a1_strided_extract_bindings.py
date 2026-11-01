#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Validate optional block-step extraction builders and IR round trips."""

from ptoas.mlir.ir import Attribute, Context, InsertionPoint, Location, Module
from ptoas.mlir.dialects import pto


def check_builder(step, rows):
    module = Module.parse(
        f"""module attributes {{pto.target_arch = "a1"}} {{
          func.func @build(%source: !pto.tile_buf<mat, 2304x32xf16, blayout=col_major, slayout=row_major, fractal=512>,
              %target: !pto.tile_buf<right, {rows}x32xf16, blayout=row_major, slayout=col_major, fractal=512>,
              %row: index, %column: index) {{ return }}
        }}"""
    )
    block = module.body.operations[0].regions[0].blocks[0]
    source, target, row, column = block.arguments
    value = None if step is None else Attribute.parse(f"{step} : i64")
    with InsertionPoint.at_block_begin(block):
        operation = pto.TExtractOp(source, row, column, target, block_step=value)
    if (
        operation.src != source
        or operation.dst != target
        or operation.block_step != value
    ):
        raise RuntimeError(
            "block-step builder changed source, destination or attribute"
        )
    if not module.operation.verify():
        raise RuntimeError("block-step builder produced invalid IR")
    restored = Module.parse(str(module))
    if not restored.operation.verify():
        raise RuntimeError("block-step extraction failed print/parse round trip")
    after = restored.body.operations[0].regions[0].blocks[0].operations[0]
    if after.block_step != value or (
        ("block_step" in str(after)) != (step is not None)
    ):
        raise RuntimeError("block_step presence or value was lost in round trip")


def main():
    with Context() as context, Location.unknown(context):
        pto.register_dialect(context, load=True)
        for step, rows in ((None, 128), (1, 128), (4, 144), (4294967295, 16)):
            check_builder(step, rows)
    print("Block-step builders, optional-one semantics and IR round trips passed.")


if __name__ == "__main__":
    main()
