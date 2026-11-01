# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Driver-free row-scatter lowering, exact CPU payloads and native objects."""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Profile:
    rows: int = 16
    cols: int = 32
    dst_rows: int = 2304
    dst_cols: int = 64
    step: int | None = 145
    row: int = 128
    column: int = 16
    kind: str = "single"
    groups: int = 1


PROFILES = [
    Profile(),
    Profile(rows=3, cols=16, dst_cols=16, column=0),
    Profile(rows=3),
    Profile(8, 16, 64, 64, 7, 14, 32),
    Profile(3, 32, 64, 64, 1, 5, 16),
    Profile(64, 16, 128, 32, 1, 31, 16),
    Profile(4096, 16, 8192, 16, 1, 4096, 0),
    Profile(4096, 16, 8192, 16, 2, 1, 0),
    Profile(16, 16, 4608, 32, 145, 2432, 16),
    Profile(16, 32, 4112, 64, 257, 256, 16),
    Profile(1, 16, 16, 16, 65536, 15, 0),
    Profile(3, 16, 32768, 16, 1, 32765, 0),
    Profile(16, 16, 32, 16, None, 16, 0),
    Profile(row=124, kind="loop"),
    Profile(row=124, kind="banks"),
    *[
        Profile(channels, columns, groups * 2304, columns, 145, 0, 0, "sparse", groups)
        for channels, groups in ((3, 1), (16, 1), (16, 2))
        for columns in (16, 32)
    ],
]


def tile(space, rows, columns):
    layout = (
        ", blayout=col_major, slayout=row_major, fractal=512" if space == "mat" else ""
    )
    return f"!pto.tile_buf<{space}, {rows}x{columns}xf16{layout}>"


class Program:
    def __init__(self, profile, level):
        self.profile = profile
        self.manual = level == "level3"
        self.constants = {}
        self.body = []
        self.source_type = tile("vec", profile.rows, profile.cols)
        self.target_type = tile("mat", profile.dst_rows, profile.dst_cols)

    def constant(self, number, dtype="index"):
        name = f"%a{number}" if dtype == "i64" else f"%c{number}"
        self.constants[(number, dtype)] = name
        return name

    def allocate(self, name, dtype, address):
        annotation = f" addr = {self.constant(address, 'i64')}" if self.manual else ""
        self.body.append(f"%{name} = pto.alloc_tile{annotation} : {dtype}")

    def load(self, name, row, col, vector):
        r, c = self.constant(self.profile.rows), self.constant(self.profile.cols)
        partition = (
            f"!pto.partition_tensor_view<{self.profile.rows}x{self.profile.cols}xf16>"
        )
        self.body.extend(
            [
                f"%{name} = pto.partition_view %iv, offsets = [{row}, {col}], sizes = [{r}, {c}] : "
                f"!pto.tensor_view<?x?xf16> -> {partition}",
                f"pto.tload ins(%{name} : {partition}) outs(%{vector} : {self.source_type})",
            ]
        )

    def insert(self, vector, row):
        attr = (
            ""
            if self.profile.step is None
            else f" {{row_step = {self.profile.step} : i64}}"
        )
        self.body.append(
            f"pto.tinsert ins(%{vector}, {row}, {self.constant(self.profile.column)} : "
            f"{self.source_type}, index, index) outs(%mat : {self.target_type}){attr}"
        )

    def render(self):
        p = self.profile
        copies = 5 if p.kind in ("loop", "banks") else 1
        stride = 9 * p.cols if p.kind == "sparse" else p.cols + 16
        rows = p.groups * p.rows if p.kind == "sparse" else copies * p.rows
        zero, one = self.constant(0), self.constant(1)
        self.body.append(
            f"%iv = pto.make_tensor_view %input, shape = [{self.constant(rows)}, "
            f"{self.constant(stride)}], strides = [{self.constant(stride)}, {one}] : !pto.tensor_view<?x?xf16>"
        )
        self.allocate(
            "mat",
            self.target_type,
            0 if p.dst_rows * p.dst_cols * 2 == 1024 * 1024 else 512,
        )
        if p.kind == "sparse":
            self.sparse(zero, one)
        else:
            self.generic(zero, one, copies)
        constants = [
            f"{name} = arith.constant {number} : {dtype}"
            for (number, dtype), name in self.constants.items()
        ]
        body = "\n    ".join([*constants, *self.body, "return"])
        return f"""module attributes {{pto.target_arch = "a1"}} {{
  func.func @rowstep_kernel(%input: !pto.ptr<f16>, %initial: !pto.ptr<f16>) attributes {{pto.entry}} {{
    {body}
  }}
}}
"""

    def generic(self, zero, one, copies):
        p = self.profile
        r, c = self.constant(p.dst_rows), self.constant(p.dst_cols)
        part = f"!pto.partition_tensor_view<{p.dst_rows}x{p.dst_cols}xf16>"
        self.body.extend(
            [
                f"%mv = pto.make_tensor_view %initial, shape = [{r}, {c}], strides = [{c}, {one}] : "
                f"!pto.tensor_view<?x?xf16>",
                f"%mp = pto.partition_view %mv, offsets = [{zero}, {zero}], sizes = [{r}, {c}] : "
                f"!pto.tensor_view<?x?xf16> -> {part}",
                f"pto.tload ins(%mp : {part}) outs(%mat : {self.target_type})",
            ]
        )
        self.allocate("source0", self.source_type, 0)
        if p.kind == "loop":
            self.body.extend(
                [
                    f"scf.for %copy = {zero} to {self.constant(copies)} step {one} {{",
                    f"%sr = arith.muli %copy, {self.constant(p.rows)} : index",
                    f"%dr = arith.addi %copy, {self.constant(p.row)} : index",
                ]
            )
            self.load("ip", "%sr", zero, "source0")
            self.insert("source0", "%dr")
            self.body.append("}")
            return
        if p.kind == "banks":
            self.allocate("source1", self.source_type, p.rows * p.cols * 2)
            self.load("ip0", zero, zero, "source0")
            self.load("ip1", self.constant(p.rows), zero, "source1")
        for copy in range(copies):
            bank = f"source{copy % 2}" if p.kind == "banks" else "source0"
            if p.kind != "banks" or copy >= 2:
                self.load(f"ip{copy}", self.constant(copy * p.rows), zero, bank)
            self.insert(bank, self.constant(p.row + copy))

    def sparse(self, zero, one):
        p = self.profile
        zero_type = tile("vec", 256, p.dst_cols)
        self.allocate("zeros", zero_type, 0)
        self.body.extend(
            [
                "%half_zero = arith.constant 0.0 : f16",
                f"pto.texpands ins(%half_zero : f16) outs(%zeros : {zero_type})",
                f"scf.for %clear = {zero} to {self.constant(p.dst_rows)} step {self.constant(256)} {{",
                f"pto.tinsert ins(%zeros, %clear, {zero} : {zero_type}, index, index) outs(%mat : {self.target_type})",
                "}",
            ]
        )
        self.allocate("source0", self.source_type, 256 * p.dst_cols * 2)
        self.body.extend(
            [
                f"scf.for %group = {zero} to {self.constant(p.groups)} step {one} {{",
                f"%gr = arith.muli %group, {self.constant(p.rows)} : index",
                f"%gd = arith.muli %group, {self.constant(2304)} : index",
                f"scf.for %tap = {zero} to {self.constant(9)} step {one} {{",
                f"%tc = arith.muli %tap, {self.constant(p.cols)} : index",
                f"%td = arith.muli %tap, {self.constant(16)} : index",
                "%dr = arith.addi %gd, %td : index",
            ]
        )
        self.load("ip", "%gr", "%tc", "source0")
        self.insert("source0", "%dr")
        self.body.extend(["}", "}"])


def static_allocations(code):
    declarations = re.findall(r"(?:u?int\d+_t|size_t) (\w+) = ([^;]+);", code)
    values = {}
    for _ in range(len(declarations)):
        for name, expression in declarations:
            expression = re.sub(r"\((?:u?int\d+_t|size_t)\)\s*", "", expression).strip()
            if re.fullmatch(r"\d+", expression):
                values[name] = int(expression)
            elif expression in values:
                values[name] = values[expression]
    return {
        variable: values.get(address)
        for variable, address in re.findall(r"TASSIGN\((\w+), (\w+)\);", code)
    }


def planned_address(code, profile):
    matrices = re.findall(
        rf"Tile<TileType::Mat, half, {profile.dst_rows}, {profile.dst_cols},[^;\n]+> (\w+);",
        code,
    )
    if len(matrices) != 1:
        raise RuntimeError(
            "generated code must retain exactly one observable resident matrix"
        )
    allocations = static_allocations(code)
    address = allocations.get(matrices[0])
    matrix_bytes = profile.dst_rows * profile.dst_cols * 2
    if (
        address is None
        or address < 0
        or address % 512
        or address + matrix_bytes > 1024 * 1024
    ):
        raise RuntimeError(
            "resident matrix address must be statically bounded within L1"
        )
    if profile.kind == "banks":
        check_banks(code, allocations, profile)
    return address


def check_banks(code, allocations, profile):
    names = re.findall(
        rf"Tile<TileType::Vec, half, {profile.rows}, {profile.cols},[^;\n]+> (\w+);",
        code,
    )
    addresses = [allocations.get(name) for name in names]
    size = profile.rows * profile.cols * 2
    if len(addresses) != 2 or any(address is None for address in addresses):
        raise RuntimeError("two live UB source banks must remain explicitly allocated")
    lower, upper = sorted(addresses)
    if (
        lower < 0
        or lower % 32
        or upper % 32
        or lower + size > upper
        or upper + size > 128 * 1024
    ):
        raise RuntimeError("live UB source banks must be aligned, disjoint and bounded")


def profile_directory(output, number, level):
    return output / f"profile{number:02d}-{level}"


def generate(output):
    for number, profile in enumerate(PROFILES):
        for level in ("level1", "level2", "level3"):
            directory = profile_directory(output, number, level)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "input.pto").write_text(Program(profile, level).render())
            print(f"{number}\t{level}\t{3 * profile.dst_rows * profile.dst_cols}")


def inspect_output(output, number, level):
    profile = PROFILES[number]
    directory = profile_directory(output, number, level)
    code = (directory / "generated.cpp").read_text()
    kernels = re.findall(r'extern "C" __global__ AICORE void (\w+)\(', code)
    if kernels != ["rowstep_kernel"]:
        raise RuntimeError("fixture must lower to exactly one original kernel")
    form = "TINSERT(" if profile.step is None else f"TINSERT<{profile.step}>("
    if form not in code or "wait_flag(PIPE_MTE2, PIPE_MTE3," not in code:
        raise RuntimeError("missing insertion form or forward DMA dependency")
    if (
        profile.kind in ("loop", "banks", "sparse")
        and "wait_flag(PIPE_MTE3, PIPE_MTE2," not in code
    ):
        raise RuntimeError("missing UB bank-reuse dependency")
    definitions = dict(
        SRC_ROWS=profile.rows,
        SRC_COLS=profile.cols,
        DST_ROWS=profile.dst_rows,
        DST_COLS=profile.dst_cols,
        ROW_STEP=profile.step or 1,
        ROW=profile.row,
        COLUMN=profile.column,
        COPIES=5 if profile.kind in ("loop", "banks") else 1,
        GROUPS=profile.groups,
        SPARSE=int(profile.kind == "sparse"),
        MAT_ADDRESS=planned_address(code, profile),
    )
    for key, value in definitions.items():
        print(f"-DPTO_TEST_ROWSTEP_{key}={value}")
    oracle = (
        ROOT / "test/npu_validation/scripts/a1_strided_insert_cpu.cpp"
    ).read_text()
    (directory / "oracle.cpp").write_text("#include <cstdint>\n" + code + "\n" + oracle)
    (directory / "native.cpp").write_text("#include <cstdint>\n" + code)


def inspect_native(output, number, level):
    directory = profile_directory(output, number, level)
    code = (directory / "native.ll").read_text()
    match = re.search(
        r"^define[^\n]*@rowstep_kernel\([^\n]*\)[^{\n]*\{\n(.*?)^\}",
        code,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError("native LLVM output must contain the original kernel body")
    copies = len(
        re.findall(r"call void @llvm\.hivm\.MOV\.UB\.TO\.L1\(", match.group(1))
    )
    if copies == 0:
        raise RuntimeError(
            "native compiler removed or failed to emit UB-to-L1 insertion"
        )
    if PROFILES[number].kind == "banks" and copies != 10:
        raise RuntimeError("five N32 inserts must retain ten native C16 copies")
    (directory / "native-summary.json").write_text(
        json.dumps(
            {
                "kernel": "rowstep_kernel",
                "static_ub_to_l1_copies": copies,
                "hardware_execution": False,
                "performance_claim": None,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "check", "native"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", type=int, choices=range(len(PROFILES)))
    parser.add_argument("--level", choices=("level1", "level2", "level3"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.mode == "generate":
        generate(output)
    elif args.profile is None or args.level is None:
        parser.error("check requires --profile and --level")
    elif args.mode == "native":
        inspect_native(output, args.profile, args.level)
    else:
        inspect_output(output, args.profile, args.level)


if __name__ == "__main__":
    main()
