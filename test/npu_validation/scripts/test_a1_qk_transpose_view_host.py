# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""A1 transposed Mat view positive and fail-closed host checks."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "test/lit/a1/qk_transpose_view.pto"


def run_ptoas(tmp_path, source):
    ptoas = shutil.which("ptoas")
    if not ptoas:
        pytest.skip("requires PTOAS")
    input_path = tmp_path / "input.pto"
    input_path.write_text(source)
    output_path = tmp_path / "output.cpp"
    result = subprocess.run(
        [
            ptoas,
            "--pto-arch=a1",
            "--pto-level=level3",
            str(input_path),
            "-o",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    return result, output_path


def test_qk_view_emits_two_accumulation_steps(tmp_path):
    result, output = run_ptoas(tmp_path, SOURCE.read_text())
    assert result.returncode == 0, result.stderr
    code = output.read_text()
    assert "TMATMUL(" in code and "TMATMUL_ACC(" in code
    for k_name, view_name in (("k0__ssa_v0", "t__tmp_v0"), ("k1__ssa_v0", "t__tmp_v1")):
        source = re.search(rf"// pto: %{k_name}\s+;?\s*Tile<[^;]+> (\w+);", code)
        view = re.search(rf"// pto: %{view_name}\s+;?\s*Tile<[^;]+> (\w+);", code)
        assert source and view
        load = re.search(rf"TLOAD\({source.group(1)},[^;]+\);", code)
        move = re.search(rf"TMOV\([^,]+, {view.group(1)}\);", code)
        assert load and move and load.end() < move.start()
        between = code[load.end() : move.start()]
        event = re.search(r"set_flag\(PIPE_MTE2, PIPE_MTE1, (EVENT_ID\d+)\);", between)
        assert (
            event and f"wait_flag(PIPE_MTE2, PIPE_MTE1, {event.group(1)});" in between
        )


@pytest.mark.parametrize("rows,location", [(16, "left"), (32, "right")])
def test_transpose_view_rejects_unsupported_move(tmp_path, rows, location):
    src = f"!pto.tile_buf<mat, {rows}x16xf16, blayout=row_major, slayout=col_major, fractal=512>"
    if location == "left":
        dst = f"!pto.tile_buf<left, {rows}x16xf16, blayout=row_major, slayout=row_major, fractal=512>"
    else:
        dst = f"!pto.tile_buf<right, {rows}x16xf16, blayout=row_major, slayout=col_major, fractal=512>"
    source = f"""module attributes {{pto.target_arch = "a1"}} {{
  func.func @bad() attributes {{pto.entry}} {{
    %zero = arith.constant 0 : i64
    %s = pto.alloc_tile addr = %zero : {src}
    %d = pto.alloc_tile addr = %zero : {dst}
    pto.tmov ins(%s : {src}) outs(%d : {dst})
    return
  }}
}}
"""
    result, _ = run_ptoas(tmp_path, source)
    assert result.returncode != 0
    assert "A1 matrix tiles require" in result.stderr


def test_transpose_view_rejects_load_as_destination(tmp_path):
    mat = "!pto.tile_buf<mat, 16x16xf16, blayout=row_major, slayout=col_major, fractal=512>"
    source = f"""module attributes {{pto.target_arch = "a1"}} {{
  func.func @bad(%input: !pto.ptr<f16>) attributes {{pto.entry}} {{
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %size = arith.constant 16 : index
    %addr = arith.constant 0 : i64
    %tensor = pto.make_tensor_view %input, shape = [%size, %size], strides = [%size, %one] : !pto.tensor_view<?x?xf16>
    %part = pto.partition_view %tensor, offsets = [%zero, %zero], sizes = [%size, %size] : !pto.tensor_view<?x?xf16> -> !pto.partition_tensor_view<16x16xf16>
    %tile = pto.alloc_tile addr = %addr : {mat}
    pto.tload ins(%part : !pto.partition_tensor_view<16x16xf16>) outs(%tile : {mat})
    return
  }}
}}
"""
    result, _ = run_ptoas(tmp_path, source)
    assert result.returncode != 0
    assert "A1 matrix tiles require" in result.stderr
