# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU oracle and offline native compilation for UB-staged convolution images."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CASES = [
    ("partial_alias", 4, 32, "a1_mat_convtile_alias"),
    ("double_buffer", 4, 32, "a1_mat_convtile_alias"),
    ("bounded_loop", 4, 80, "a1_mat_convtile_alias"),
    ("40x40_double_buffer", 40, 64, "a1_im2col_40x40"),
]


@pytest.fixture(params=CASES, ids=[case[0] for case in CASES])
def generated_case(request, tmp_path):
    ptoas = shutil.which("ptoas")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not ptoas or not isa:
        pytest.skip("requires PTOAS and a matching PTO_ISA_ROOT")
    case, side, channels, kernel = request.param
    generated = tmp_path / "generated.cpp"
    subprocess.run(
        [
            ptoas,
            "--pto-arch=a1",
            "--pto-level=level3",
            str(ROOT / f"test/lit/a1/vec_mat_convtile_{case}.pto"),
            "-o",
            str(generated),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    code = generated.read_text()
    assert "set_flag(PIPE_MTE3, PIPE_MTE1," in code
    assert "wait_flag(PIPE_MTE3, PIPE_MTE1," in code
    assert "TIMG2COL<" in code
    assert code.count("TINSERT(") == (
        1 if case == "bounded_loop" else 25 if side == 40 else 2
    )
    if case == "bounded_loop":
        assert "for (" in code
        assert "wait_flag(PIPE_MTE3, PIPE_MTE2," in code
    elif case == "partial_alias":
        assert "wait_flag(PIPE_MTE3, PIPE_MTE2," in code
    elif side == 40:
        assert code.count("wait_flag(PIPE_MTE3, PIPE_MTE2,") >= 23
    else:
        assert code.count("set_flag(PIPE_MTE2, PIPE_MTE3,") >= 2
    return generated, Path(isa), side, channels, kernel


def test_staged_image_cpu_oracle(tmp_path, generated_case):
    compiler = shutil.which("clang++-19")
    if not compiler:
        pytest.skip("requires clang++-19")
    generated, isa, side, channels, kernel = generated_case
    cpu = tmp_path / "cpu.cpp"
    cpu.write_text(
        "#include <cstdint>\n"
        + generated.read_text()
        + "\n"
        + (ROOT / "test/npu_validation/scripts/a1_vec_mat_convtile_cpu.cpp").read_text()
    )
    binary = tmp_path / "cpu-oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            "-D__DAV_VEC__",
            f"-DSIDE={side}",
            f"-DCHANNELS={channels}",
            f"-DKERNEL={kernel}",
            f"-I{isa / 'include'}",
            str(cpu),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_staged_image_native_compile(tmp_path, generated_case):
    compiler = shutil.which("ccec")
    if not compiler:
        pytest.skip("requires CCEC for offline A1 compilation")
    generated, isa, _, _, _ = generated_case
    subprocess.run(
        [
            compiler,
            "-c",
            "-O2",
            "-x",
            "cce",
            "-std=c++17",
            "--cce-aicore-only",
            "--cce-aicore-arch=dav-m200",
            f"-I{isa / 'include'}",
            str(generated),
            "-o",
            str(tmp_path / "native.o"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "source_type,destination_type,row,column,accepted",
    [
        (
            "!pto.tile_buf<vec, 16x16xf16>",
            "!pto.tile_buf<mat, 64x64xf16, blayout=col_major, slayout=row_major, fractal=512>",
            48,
            48,
            True,
        ),
        (
            "!pto.tile_buf<vec, 32x32xf16>",
            "!pto.tile_buf<mat, 64x64xf16, blayout=col_major, slayout=row_major, fractal=512>",
            32,
            32,
            True,
        ),
        (
            "!pto.tile_buf<vec, 32x32xf32>",
            "!pto.tile_buf<mat, 64x64xf16, blayout=col_major, slayout=row_major, fractal=512>",
            0,
            0,
            False,
        ),
        (
            "!pto.tile_buf<vec, 32x32xf16>",
            "!pto.tile_buf<mat, 64x64xf16, blayout=row_major, slayout=row_major, fractal=512>",
            0,
            0,
            False,
        ),
        (
            "!pto.tile_buf<vec, 8x32xf16>",
            "!pto.tile_buf<mat, 64x64xf16, blayout=col_major, slayout=row_major, fractal=512>",
            0,
            0,
            False,
        ),
        (
            "!pto.tile_buf<vec, 32x96xf16>",
            "!pto.tile_buf<mat, 64x64xf16, blayout=col_major, slayout=row_major, fractal=512>",
            0,
            0,
            False,
        ),
        (
            "!pto.tile_buf<vec, 32x32xf16>",
            "!pto.tile_buf<mat, 64x64xf16, blayout=col_major, slayout=row_major, fractal=512>",
            -16,
            0,
            False,
        ),
        (
            "!pto.tile_buf<vec, 32x32xf16>",
            "!pto.tile_buf<mat, 64x64xf16, blayout=col_major, slayout=row_major, fractal=512>",
            0,
            8,
            False,
        ),
        (
            "!pto.tile_buf<vec, 32x32xf16>",
            "!pto.tile_buf<mat, 64x64xf16, blayout=col_major, slayout=row_major, fractal=512>",
            48,
            0,
            False,
        ),
        (
            "!pto.tile_buf<vec, 32x32xf16>",
            "!pto.tile_buf<mat, 64x64xf16, blayout=col_major, slayout=row_major, fractal=512>",
            None,
            0,
            False,
        ),
    ],
    ids=[
        "c16-last-window",
        "wide-last-window",
        "fp32-source",
        "zz-destination",
        "short-source",
        "wide-source",
        "negative-row",
        "unaligned-column",
        "out-of-bounds",
        "unbounded-row",
    ],
)
def test_insert_contract(
    tmp_path, source_type, destination_type, row, column, accepted
):
    ptoas = shutil.which("ptoas")
    if not ptoas:
        pytest.skip("requires PTOAS")
    row_value = "%unknown" if row is None else "%row"
    row_constant = "" if row is None else f"%row = arith.constant {row} : index"
    source = tmp_path / "contract.pto"
    source.write_text(f"""module attributes {{pto.target_arch = "a1"}} {{
  func.func @contract(%unknown: index) attributes {{pto.entry}} {{
    %address = arith.constant 0 : i64
    {row_constant}
    %column = arith.constant {column} : index
    %source = pto.alloc_tile addr = %address : {source_type}
    %destination = pto.alloc_tile addr = %address : {destination_type}
    pto.tinsert ins(%source, {row_value}, %column : {source_type}, index, index)
      outs(%destination : {destination_type})
    return
  }}
}}
""")
    result = subprocess.run(
        [
            ptoas,
            "--pto-arch=a1",
            "--pto-level=level3",
            str(source),
            "-o",
            str(tmp_path / "contract.cpp"),
        ],
        capture_output=True,
        text=True,
    )
    if accepted:
        assert result.returncode == 0, result.stderr
        assert "TINSERT(" in (tmp_path / "contract.cpp").read_text()
    else:
        assert result.returncode != 0
        assert "error:" in result.stderr
        if "32x96" in source_type:
            assert "indexCol + src.cols <= dst.cols" in result.stderr
        elif row is not None and row < 0:
            assert "indexRow to be non-negative" in result.stderr
        elif row == 48:
            assert "indexRow + src.rows <= dst.rows" in result.stderr
        else:
            assert "A1" in result.stderr
