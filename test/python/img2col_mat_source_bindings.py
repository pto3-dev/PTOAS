#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check Mat-source im2col builders without opening any device runtime."""

from ptoas.mlir.ir import Attribute, Context, InsertionPoint, Location, Module
from ptoas.mlir.dialects import pto


CONFIG = """#pto.conv_tile_config<fmap_h=4, fmap_w=4, pad_list=1x1x1x1,
    filter_h=3, filter_w=3, dilation_h=1, dilation_w=1, stride_h=1, stride_w=1,
    pad_value=0 : i32, channel_size=32, repeat_stride=0, repeat_time=1,
    repeat_mode=0, dst_stride=0, dst_mposition=0, transpose=false>"""


def check_builder(
    source_type: str, config, has_m: bool, has_k: bool, filter_order=None
) -> None:
    columns = 288 if filter_order is None else 16 * len(filter_order)
    selection = (
        None
        if filter_order is None
        else Attribute.parse("array<i64: " + ", ".join(map(str, filter_order)) + ">")
    )
    module = Module.parse(
        """module attributes {pto.target_arch = "a1"} {
      func.func @build(%image: SOURCE,
          %left: !pto.tile_buf<left, 16x288xf16, blayout=row_major, slayout=row_major, fractal=512>,
          %m: index, %k: index) {
        return
      }
    }""".replace("SOURCE", source_type).replace("16x288", f"16x{columns}")
    )
    block = module.body.operations[0].regions[0].blocks[0]
    image, left, m, k = block.arguments
    expected_m = m if has_m else None
    expected_k = k if has_k else None
    with InsertionPoint.at_block_begin(block):
        op = pto.TImg2colOp(
            left,
            image,
            posM=expected_m,
            posK=expected_k,
            fmatrixMode=Attribute.parse("#pto.fmatrix_mode<a_auto>"),
            config=config,
            filter_order=selection,
        )
    if (
        op.src != image
        or op.dst != left
        or op.posM != expected_m
        or op.posK != expected_k
        or op.config != config
        or op.filter_order != selection
    ):
        raise RuntimeError(
            "im2col builder lost source identity, position operands or configuration"
        )
    if not module.operation.verify():
        raise RuntimeError("im2col builder produced invalid IR")
    restored = Module.parse(str(module))
    if not restored.operation.verify():
        raise RuntimeError("im2col builder failed the print/parse round trip")
    after = restored.body.operations[0].regions[0].blocks[0].operations[0]
    restored_args = restored.body.operations[0].regions[0].blocks[0].arguments
    restored_m = restored_args[2] if has_m else None
    restored_k = restored_args[3] if has_k else None
    if (
        after.config != config
        or after.posM != restored_m
        or after.posK != restored_k
        or after.filter_order != selection
    ):
        raise RuntimeError("im2col round trip changed geometry or optional positions")


def main() -> None:
    with Context() as context, Location.unknown(context):
        pto.register_dialect(context, load=True)
        for has_m, has_k in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            for selection in (
                None,
                (0, 4, 8, 3, 7, 2, 6, 1, 5),
                (0, 4, 8, 3, 7, 2),
                (6, 7, 8),
            ):
                check_builder(
                    "!pto.tile_buf<mat, 16x32xf16, blayout=col_major, slayout=row_major, fractal=512>",
                    Attribute.parse(CONFIG),
                    has_m,
                    has_k,
                    selection,
                )
                # The old source form and its omitted operation config remain valid.
                check_builder(
                    "!pto.conv_tile_buf<mat, 1x2x4x4x16xf16, layout=#pto.conv_layout<nc1hwc0>>",
                    None,
                    has_m,
                    has_k,
                    selection,
                )
    print("img2col_mat_source_bindings: PASS")


if __name__ == "__main__":
    main()
