# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host checks for Q/K panel ordering and synchronized row reuse on A1."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "test/lit/a1/qk_row_reuse.pto"


def check_qk_sync(code):
    def tile(name):
        match = re.search(rf"// pto: %{name}\s+;?\s*Tile<[^;]+> (\w+);", code)
        assert match, f"missing tile {name}"
        return match.group(1)

    def load(name):
        match = re.search(rf"TLOAD\({tile(name)},[^;]+\);", code)
        assert match, f"missing load {name}"
        return match

    def move(src):
        match = re.search(rf"TMOV\([^,]+, {tile(src)}\);", code)
        assert match, f"missing move from {src}"
        return match

    q0, q1 = load("q0__ssa_v0"), load("q1__ssa_v0")
    k0, k1 = load("k0__ssa_v0"), load("k1__ssa_v0")
    assert q0.end() < q1.start() < k0.start() < k1.start()
    for producer, consumer in (
        (q0, move("q0__ssa_v0")),
        (q1, move("q1__ssa_v0")),
        (k0, move("t__tmp_v2")),
        (k1, move("t__tmp_v3")),
    ):
        assert producer.end() < consumer.start()
        between = code[producer.end() : consumer.start()]
        events = re.findall(
            r"set_flag\(PIPE_MTE2, PIPE_MTE1, (EVENT_ID\d+)\);", between
        )
        assert any(
            f"wait_flag(PIPE_MTE2, PIPE_MTE1, {event});" in between for event in events
        )

    # Before replacing the Q row at the next outer iteration, MTE2 must wait
    # for MTE1 to finish reading it. The first iteration uses the initial token.
    before_q = code[: q0.start()]
    waits = re.findall(r"wait_flag\(PIPE_MTE1, PIPE_MTE2, (EVENT_ID\d+)\);", before_q)
    after_q1_move = code[move("q1__ssa_v0").end() :].split("\n  }", 1)[0]
    assert any(
        f"set_flag(PIPE_MTE1, PIPE_MTE2, {event});" in after_q1_move for event in waits
    )
    assert "TMATMUL(" in code and "TMATMUL_ACC(" in code


def test_qk_row_reuse_sync(tmp_path):
    ptoas = shutil.which("ptoas")
    if not ptoas:
        pytest.skip("requires PTOAS")
    output = tmp_path / "out.cpp"
    result = subprocess.run(
        [ptoas, "--pto-arch=a1", "--pto-level=level3", str(SOURCE), "-o", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    check_qk_sync(output.read_text())


def test_qk_row_reuse_generated_cpu(tmp_path):
    import os  # noqa: PLC0415

    ptoas, compiler = shutil.which("ptoas"), shutil.which("clang++-19")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not ptoas or not compiler or not isa:
        pytest.skip("requires PTOAS, clang++-19 and PTO_ISA_ROOT")
    generated = tmp_path / "generated.cpp"
    subprocess.run(
        [
            ptoas,
            "--pto-arch=a1",
            "--pto-level=level3",
            str(SOURCE),
            "-o",
            str(generated),
        ],
        check=True,
    )
    binary = tmp_path / "oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-ffp-contract=off",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            f"-I{isa}/include",
            "-include",
            "cstdint",
            "-include",
            str(generated),
            str(ROOT / "test/npu_validation/scripts/a1_qk_row_reuse_cpu.cpp"),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for cores in (1, 3, 8, 32):
        subprocess.run(
            [str(binary), str(cores)],
            check=True,
            env={
                **os.environ,
                "PTO_CPU_SIM_NUM_CORES": str(cores),
                "PTO_CPU_SIM_TRACE_ENABLE": "0",
            },
        )
