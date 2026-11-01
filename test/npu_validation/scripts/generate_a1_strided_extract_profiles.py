# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"Compile existing TEXTRACT through all planners; exact CPU and native checks."

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

from generate_a1_strided_insert_profiles import Profile as InsertProfile
from generate_a1_strided_insert_profiles import Program as InsertProgram
from generate_a1_strided_insert_profiles import static_allocations

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Profile:
    source_rows: int = 2304
    source_cols: int = 32
    target_rows: int = 144
    target_cols: int = 32
    step: int | None = 4
    row: int = 1776
    column: int = 0
    zz: bool = False
    channels: int = 0
    groups: int = 1


PROFILES = (
    Profile(),
    Profile(4608, 64, 144, 32, 4, 4080, 16, True),
    Profile(32768, 16, 32, 16, 2047, 0),
    Profile(256, 2048, 32, 32, 1, 224, 2016, True),
    Profile(16, 16, 16, 16, 4294967295, 0),
    Profile(256, 64, 128, 32, None, 128, 32),
    Profile(256, 64, 128, 32, 1, 128, 32, True),
    Profile(4096, 16, 2048, 16, 2, 16),
    Profile(2048, 32, 1024, 32, 2, 16, 0, True),
    Profile(2304, 32, 144, 16, 4, 0, 16, True),
    *[
        Profile(groups * 2304, columns, 144, columns, 4, 0, 0, False, channels, groups)
        for channels, groups in ((3, 1), (16, 1), (16, 2), (16, 4))
        for columns in (16, 32)
    ],
)


def tile(space, rows, columns, zz=False):
    outer = "row_major" if zz or space in ("left", "right") else "col_major"
    inner = "col_major" if space == "right" else "row_major"
    dtype, fractal = ("f32", 1024) if space == "acc" else ("f16", 512)
    return f"!pto.tile_buf<{space}, {rows}x{columns}x{dtype}, blayout={outer}, slayout={inner}, fractal={fractal}>"


class CubeProgram(InsertProgram):
    "Reuse the compact-weight insertion fixture, then consume both Right banks."

    def sparse(self, zero, one):
        super().sparse(zero, one)
        left_mat = tile("mat", 16, 144, True)
        left = tile("left", 16, 144)
        right = tile("right", 144, self.profile.cols)
        acc = tile("acc", 16, self.profile.cols)
        # The streamed production cache holds two banks of two C16 groups.
        # Place Left storage after the resident panel when four groups exceed 512 KiB.
        left_address = max(
            524288, 512 + self.profile.dst_rows * self.profile.dst_cols * 2
        )
        self.allocate("left_mat", left_mat, left_address)
        self.allocate("left", left, 0)
        self.allocate("right0", right, 0)
        self.allocate("right1", right, 16384)
        self.allocate("acc", acc, 0)
        m, k = self.constant(16), self.constant(144)
        partition = "!pto.partition_tensor_view<16x144xf16>"
        self.body.extend(
            [
                f"%lv = pto.make_tensor_view %initial, shape = [{m}, {k}], strides = [{k}, {one}] : "
                "!pto.tensor_view<?x?xf16>",
                f"%lp = pto.partition_view %lv, offsets = [{zero}, {zero}], sizes = [{m}, {k}] : "
                f"!pto.tensor_view<?x?xf16> -> {partition}",
                f"pto.tload ins(%lp : {partition}) outs(%left_mat : {left_mat})",
                f"pto.tmov ins(%left_mat : {left_mat}) outs(%left : {left})",
            ]
        )
        source = self.target_type
        rows = (0, 16, 32, 48, (self.profile.groups - 1) * 2304 + 576)

        def extract(index):
            bank = index % 2
            self.body.append(
                f"pto.textract ins(%mat, {self.constant(rows[index])}, {zero} : {source}, index, index) "
                f"outs(%right{bank} : {right}) {{block_step = 4 : i64}}"
            )

        extract(0)
        extract(1)
        self.body.append(
            f"pto.tmatmul ins(%left, %right0 : {left}, {right}) outs(%acc : {acc})"
        )
        for index in range(1, 5):
            if index >= 2:
                extract(index)
            self.body.append(
                f"pto.tmatmul.acc ins(%acc, %left, %right{index % 2} : {acc}, {left}, {right}) "
                f"outs(%acc : {acc})"
            )


def program(profile, level):
    if profile.channels:
        config = InsertProfile(
            profile.channels,
            profile.target_cols,
            profile.source_rows,
            profile.source_cols,
            145,
            0,
            0,
            "sparse",
            profile.groups,
        )
        return CubeProgram(config, level).render(), "rowstep_kernel"
    source = tile("mat", profile.source_rows, profile.source_cols, profile.zz)
    target = tile("right", profile.target_rows, profile.target_cols)
    matrix_bytes = profile.source_rows * profile.source_cols * 2
    target_bytes = profile.target_rows * profile.target_cols * 2
    manual = level == "level3"
    source_addr = " addr = %source_addr" if manual else ""
    target_addr = " addr = %target_addr" if manual else ""
    attr = "" if profile.step is None else f" {{block_step = {profile.step} : i64}}"
    partition = (
        f"!pto.partition_tensor_view<{profile.source_rows}x{profile.source_cols}xf16>"
    )
    return (
        f"""module attributes {{pto.target_arch = "a1"}} {{
  func.func @extract_kernel(%input: !pto.ptr<f16>, %initial: !pto.ptr<f16>) attributes {{pto.entry}} {{
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %sixteen = arith.constant 16 : index
    %rows = arith.constant {profile.source_rows} : index
    %cols = arith.constant {profile.source_cols} : index
    %stride = arith.constant {profile.source_cols + 16} : index
    %row = arith.constant {profile.row} : index
    %col = arith.constant {profile.column} : index
    %source_addr = arith.constant {0 if matrix_bytes == 1048576 else 512} : i64
    %target_addr = arith.constant {0 if target_bytes == 65536 else 512} : i64
    %source = pto.alloc_tile{source_addr} : {source}
    %target = pto.alloc_tile{target_addr} : {target}
    %view = pto.make_tensor_view %input, shape = [%rows, %stride], strides = [%stride, %one] : !pto.tensor_view<?x?xf16>
    %panel = pto.partition_view %view, offsets = [%zero, %sixteen], sizes = [%rows, %cols] :
        !pto.tensor_view<?x?xf16> -> {partition}
    pto.tload ins(%panel : {partition}) outs(%source : {source})
    pto.textract ins(%source, %row, %col : {source}, index, index) outs(%target : {target}){attr}
    return
  }}
}}
""",
        "extract_kernel",
    )


def allocation(code, addresses, space, rows, columns, dtype):
    names = re.findall(
        rf"Tile<TileType::{space}, {dtype}, {rows}, {columns},[^;\n]+> (\w+);", code
    )
    if len(names) != 1 or addresses.get(names[0]) is None:
        raise RuntimeError(f"Expected one bounded {space} allocation")
    return addresses[names[0]]


def definitions(code, profile, kernel):
    kernels = re.findall(r'extern "C" __global__ AICORE void (\w+)\(', code)
    if kernels != [kernel]:
        raise RuntimeError("Expected exactly the original kernel")
    addresses = static_allocations(code)
    source = allocation(
        code, addresses, "Mat", profile.source_rows, profile.source_cols, "half"
    )
    destinations = re.findall(r"TEXTRACT(?:<\d+>)?\((\w+),", code)
    expected_calls = 5 if profile.channels else 1
    if len(destinations) != expected_calls:
        raise RuntimeError("Lost or duplicated an extraction")
    if profile.channels and destinations != [
        destinations[0],
        destinations[1],
        destinations[0],
        destinations[1],
        destinations[0],
    ]:
        raise RuntimeError("Right bank reuse changed")
    right0 = addresses.get(destinations[0])
    right1 = addresses.get(destinations[1]) if profile.channels else right0
    size = profile.target_rows * profile.target_cols * 2
    if (
        right0 is None
        or right1 is None
        or any(a % 512 or a < 0 or a + size > 65536 for a in (right0, right1))
    ):
        raise RuntimeError("Right allocation exceeds L0B or alignment")
    if profile.channels and abs(right0 - right1) < size:
        raise RuntimeError("Simultaneously live Right banks overlap")
    if (
        source < 0
        or source % 512
        or source + profile.source_rows * profile.source_cols * 2 > 1048576
    ):
        raise RuntimeError("Resident allocation exceeds L1 or alignment")
    call = "TEXTRACT(" if profile.step is None else f"TEXTRACT<{profile.step}>("
    if call not in code:
        raise RuntimeError("Lost block_step presence or value")
    if "wait_flag(PIPE_MTE2, PIPE_MTE1," not in code and not profile.channels:
        raise RuntimeError("Missing source load readiness")
    if profile.channels:
        for pair in (("MTE3", "MTE1"), ("MTE1", "M"), ("M", "MTE1")):
            if f"wait_flag(PIPE_{pair[0]}, PIPE_{pair[1]}," not in code:
                raise RuntimeError(f"Missing compact-weight/Cube dependency {pair}")
    values = dict(
        SOURCE_ROWS=profile.source_rows,
        SOURCE_COLS=profile.source_cols,
        TARGET_ROWS=profile.target_rows,
        TARGET_COLS=profile.target_cols,
        STEP=profile.step or 1,
        ROW=profile.row,
        COLUMN=profile.column,
        ZZ=int(profile.zz),
        CHANNELS=profile.channels,
        GROUPS=profile.groups,
        SOURCE_ADDRESS=source,
        RIGHT0_ADDRESS=right0,
        RIGHT1_ADDRESS=right1,
        LEFT_MAT_ADDRESS=allocation(code, addresses, "Mat", 16, 144, "half")
        if profile.channels
        else 0,
        ACC_ADDRESS=allocation(code, addresses, "Acc", 16, profile.target_cols, "float")
        if profile.channels
        else 0,
    )
    return [f"-DPTO_TEST_EXTRACT_{key}={value}" for key, value in values.items()]


def check_native(code, kernel, profile):
    match = re.search(
        rf"^define[^\n]*@{kernel}\([^\n]*\)[^{{\n]*\{{\n(.*?)^\}}",
        code,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise RuntimeError("Missing original native kernel")
    body = match.group(1)
    calls = re.findall(
        r"call void @llvm\.hivm\.LOAD\.L1\.TO\.L0B\.2D\.f16\.addr\([^\n]+?i64 (\d+), i64 (\d+), i64 (\d+)\)",
        body,
    )
    expected = (5 if profile.channels else 1) * (profile.target_cols // 16)
    if len(calls) != expected:
        raise RuntimeError("Native Right panel-load count changed")
    repeat = profile.target_rows // 16
    step = (
        1
        if repeat == 1
        else (profile.step or 1) * (profile.source_cols // 16 if profile.zz else 1)
    )
    wanted = (repeat << 16 | step << 24 | (profile.target_cols // 16 - 1) << 44, 1, 0)
    if any(tuple(map(int, call)) != wanted for call in calls):
        raise RuntimeError(
            "Native source step, transpose or destination stride changed"
        )
    if profile.channels:
        if len(re.findall(r"call void @llvm\.hivm\.MAD\.f162f32", body)) != 5:
            raise RuntimeError("Cube consumer lost a product")
        if "call void @llvm.hivm.MOV.UB.TO.L1(" not in body:
            raise RuntimeError(
                "Compact weights no longer use public UB-to-L1 insertion"
            )
    return len(calls)


def directory_for(output, number, level):
    return output / f"profile{number:02d}-{level}"


def generate(output, numbers, levels):
    for number in numbers:
        for level in levels:
            directory = directory_for(output, number, level)
            directory.mkdir(parents=True, exist_ok=True)
            source, _ = program(PROFILES[number], level)
            (directory / "input.pto").write_text(source)
            print(f"{number}\t{level}")


def inspect_output(output, number, level):
    directory = directory_for(output, number, level)
    profile = PROFILES[number]
    _, kernel = program(profile, level)
    code = (directory / "generated.cpp").read_text()
    macros = definitions(code, profile, kernel)
    (directory / "definitions.json").write_text(json.dumps(macros, indent=2) + "\n")
    oracle = (
        ROOT / "test/npu_validation/scripts/a1_strided_extract_cpu.cpp"
    ).read_text()
    (directory / "oracle.cpp").write_text("#include <cstdint>\n" + code + "\n" + oracle)
    (directory / "native.cpp").write_text("#include <cstdint>\n" + code)
    print("\n".join(macros))


def inspect_native(output, number, level):
    directory = directory_for(output, number, level)
    profile = PROFILES[number]
    _, kernel = program(profile, level)
    panels = check_native((directory / "native.ll").read_text(), kernel, profile)
    summary = dict(
        level=level,
        profile=profile.__dict__,
        native_panels=panels,
        original_kernel=kernel,
        device_execution=False,
        performance_claim=None,
    )
    (directory / "native-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("generate", "check", "native"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=int, choices=range(len(PROFILES)))
    parser.add_argument("--level", choices=("level1", "level2", "level3"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.mode == "generate":
        numbers = range(len(PROFILES)) if args.profile is None else [args.profile]
        levels = ("level1", "level2", "level3") if args.level is None else [args.level]
        generate(output, numbers, levels)
    elif args.profile is None or args.level is None:
        parser.error("check/native require --profile and --level")
    elif args.mode == "check":
        inspect_output(output, args.profile, args.level)
    else:
        inspect_native(output, args.profile, args.level)


if __name__ == "__main__":
    main()
