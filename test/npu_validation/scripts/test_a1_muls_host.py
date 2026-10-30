# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compile emitted scalar multiply and run a bitwise CPU oracle, without NPU access."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CASES = [
    ("vector_f32", "a1_f32_muls", False),
    ("vector_f16", "a1_f16_muls", True),
    ("exact_alias", "a1_f32_muls", False),
]


@pytest.fixture(params=CASES, ids=[case[0] for case in CASES])
def generated_case(request, tmp_path):
    compiler = shutil.which("ptoas")
    isa = os.environ.get("PTO_ISA_ROOT")
    if not compiler or not isa:
        pytest.skip("requires PTOAS and a matching PTO_ISA_ROOT")
    case, kernel, fp16 = request.param
    generated = tmp_path / "generated.cpp"
    subprocess.run(
        [
            compiler,
            "--pto-arch=a1",
            str(ROOT / f"test/lit/a1/muls_{case}.pto"),
            "-o",
            str(generated),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    code = generated.read_text()
    assert "TMULS(" in code
    assert "set_flag(PIPE_MTE2, PIPE_V," in code
    assert "wait_flag(PIPE_V, PIPE_MTE3," in code
    assert f"void {kernel}(" in code
    return generated, Path(isa), kernel, fp16


def test_muls_generated_cpu_oracle(tmp_path, generated_case):
    compiler = shutil.which("clang++-19")
    if not compiler:
        pytest.skip("requires clang++-19")
    generated, isa, kernel, fp16 = generated_case
    source = tmp_path / "oracle.cpp"
    source.write_text(
        "#include <cstdint>\n"
        + generated.read_text()
        + "\n"
        + (ROOT / "test/npu_validation/scripts/a1_muls_cpu.cpp").read_text()
    )
    binary = tmp_path / "oracle"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-ffp-contract=off",
            "-D__CPU_SIM",
            f"-DKERNEL={kernel}",
            *(["-DMULS_FP16"] if fp16 else []),
            f"-I{isa / 'include'}",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)


def test_muls_generated_native_compile(tmp_path, generated_case):
    compiler = shutil.which("ccec")
    if not compiler:
        pytest.skip("requires CCEC for offline A1 compilation")
    generated, isa, _, _ = generated_case
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
