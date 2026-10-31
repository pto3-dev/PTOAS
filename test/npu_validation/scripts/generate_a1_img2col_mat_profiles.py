#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate and inspect IR fixtures for the driver-free Mat im2col shell test."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Profile:
    name: str
    h: int = 4
    w: int = 4
    c: int = 32
    rows: int = 16
    k: int = 288
    m0: int = 0
    k0: int = 0
    kh: int = 3
    kw: int = 3
    sh: int = 1
    sw: int = 1
    dh: int = 1
    dw: int = 1
    pl: int = 1
    pr: int = 1
    pt: int = 1
    pb: int = 1
    images: int = 1
    sum_images: bool = False
    ub_stage: bool = False
    omit_m: bool = False

    def macros(self) -> list[str]:
        names = (
            "h",
            "w",
            "c",
            "rows",
            "k",
            "m0",
            "k0",
            "kh",
            "kw",
            "sh",
            "sw",
            "dh",
            "dw",
            "pl",
            "pr",
            "pt",
            "pb",
            "images",
        )
        return [f"-DIMG_{name.upper()}={getattr(self, name)}" for name in names] + [
            f"-DIMG_SUM={int(self.sum_images)}"
        ]


PROFILES = (
    Profile("two_c16"),
    Profile("one_c16", c=16, k=144),
    Profile("nonzero_windows", w=8, c=16, k=16, m0=16, k0=16),
    Profile("asymmetric", w=8, c=16, k=96, kh=2, sw=2, pr=2, pt=0),
    Profile("m224", h=2, w=448, c=16, rows=224, k=144, sh=2, sw=2),
    Profile("m2048", h=32, w=64, c=16, rows=2048, k=16, k0=64),
    Profile("five_reloads", images=5),
    Profile("two_live_images", images=2, sum_images=True),
    Profile("ub_five_reloads", images=5, ub_stage=True),
    Profile("ub_two_live_images", images=2, sum_images=True, ub_stage=True),
    Profile("k2048_many_c16", c=256, k=2048, k0=256),
    Profile("c2048_last_plane", c=2048, k=16, k0=18416, omit_m=True),
)


def tile(location: str, rows: int, columns: int, element: str = "f16") -> str:
    layouts = {
        "mat": ("col_major", "row_major", 512),
        "left": ("row_major", "row_major", 512),
        "right": ("row_major", "col_major", 512),
        "acc": ("col_major", "row_major", 1024),
    }
    layout = layouts.get(location)
    suffix = (
        ""
        if layout is None
        else f", blayout={layout[0]}, slayout={layout[1]}, fractal={layout[2]}"
    )
    return f"!pto.tile_buf<{location}, {rows}x{columns}x{element}{suffix}>"


def make_program(p: Profile, explicit: bool) -> str:
    pixels = p.h * p.w
    outputs = p.rows * (1 if p.sum_images else p.images)
    count = 2 if p.sum_images else 1
    image_bytes = pixels * p.c * 2
    aligned_image = (image_bytes + 511) // 512 * 512
    shape = {
        "image": tile("mat", pixels, p.c),
        "weights": tile("mat", p.k, 16),
        "left": tile("left", p.rows, p.k),
        "right": tile("right", p.k, 16),
        "acc": tile("acc", p.rows, 16, "f32"),
        "result": tile("vec", p.rows, 16, "f32"),
    }
    shape["image_second"] = shape["image"]
    shape["left_second"] = shape["left"]
    buffers = ["image", "weights", "left", "right", "acc", "result"]
    if p.sum_images:
        buffers += ["image_second", "left_second"]
    if p.ub_stage:
        for bank in range(2):
            name = f"staging{bank}"
            shape[name] = tile("vec", pixels, 16)
            buffers.append(name)
    addresses = {
        "image": 0,
        "image_second": aligned_image,
        "weights": aligned_image * count,
        "left": 0,
        "left_second": p.rows * p.k * 2,
        "right": 0,
        "acc": 0,
        "result": 0,
    }
    # The output Vec is written after both UB staging banks are drained.
    addresses.update(staging0=0, staging1=pixels * 16 * 2)
    constants = sorted(
        {
            0,
            1,
            16,
            pixels,
            p.c,
            p.k,
            p.rows,
            p.m0,
            p.k0,
            p.images,
            pixels * p.images,
            outputs,
        }
    )
    if p.ub_stage:
        constants = sorted(set(constants) | set(range(0, p.c, 16)))
    lines = [f"    %c{n} = arith.constant {n} : index" for n in constants]
    for name in buffers:
        address = ""
        if explicit:
            lines.append(f"    %addr_{name} = arith.constant {addresses[name]} : i64")
            address = f" addr = %addr_{name}"
        lines.append(f"    %{name} = pto.alloc_tile{address} : {shape[name]}")
    for name, pointer, height, width in (
        ("iv", "input", pixels * p.images, p.c),
        ("wv", "weight", p.k, 16),
        ("ov", "out", outputs, 16),
    ):
        lines.append(
            f"    %{name} = pto.make_tensor_view %{pointer}, shape = [%c{height}, %c{width}], "
            f"strides = [%c{width}, %c1] : !pto.tensor_view<?x?x{'f32' if name == 'ov' else 'f16'}>"
        )
    lines += [
        f"    %wp = pto.partition_view %wv, offsets = [%c0, %c0], sizes = [%c{p.k}, %c16] : "
        f"!pto.tensor_view<?x?xf16> -> !pto.partition_tensor_view<{p.k}x16xf16>",
        f"    pto.tload ins(%wp : !pto.partition_tensor_view<{p.k}x16xf16>) outs(%weights : {shape['weights']})",
        f"    pto.tmov ins(%weights : {shape['weights']}) outs(%right : {shape['right']})",
    ]
    loop = p.images > 1 and not p.sum_images
    if loop:
        lines += [
            f"    scf.for %step = %c0 to %c{p.images} step %c1 {{",
            f"      %input_offset = arith.muli %step, %c{pixels} : index",
            f"      %output_offset = arith.muli %step, %c{p.rows} : index",
        ]
    for image in range(count):
        suffix = "" if image == 0 else "_second"
        offset = "%input_offset" if loop else f"%c{pixels if image else 0}"
        if not p.ub_stage:
            lines += [
                f"      %ip{image} = pto.partition_view %iv, offsets = [{offset}, %c0], "
                f"sizes = [%c{pixels}, %c{p.c}] : !pto.tensor_view<?x?xf16> -> "
                f"!pto.partition_tensor_view<{pixels}x{p.c}xf16>",
                f"      pto.tload ins(%ip{image} : !pto.partition_tensor_view<{pixels}x{p.c}xf16>) "
                f"outs(%image{suffix} : {shape['image']})",
            ]
            continue
        for group in range(0, p.c, 32):
            channels = list(enumerate(range(group, min(group + 32, p.c), 16)))
            for bank, channel in channels:
                name = f"ip{image}_{channel}"
                lines += [
                    f"      %{name} = pto.partition_view %iv, offsets = [{offset}, %c{channel}], "
                    f"sizes = [%c{pixels}, %c16] : !pto.tensor_view<?x?xf16> -> "
                    f"!pto.partition_tensor_view<{pixels}x16xf16>",
                    f"      pto.tload ins(%{name} : !pto.partition_tensor_view<{pixels}x16xf16>) "
                    f"outs(%staging{bank} : {shape[f'staging{bank}']})",
                ]
            for bank, channel in channels:
                lines.append(
                    f"      pto.tinsert ins(%staging{bank}, %c0, %c{channel} : "
                    f"{shape[f'staging{bank}']}, index, index) outs(%image{suffix} : {shape['image']})"
                )
    for image in range(count):
        suffix = "" if image == 0 else "_second"
        left = "%left" + suffix
        positions = f", pos_k = %c{p.k0}"
        if not p.omit_m:
            positions = f", pos_m = %c{p.m0}" + positions
        lines.append(
            f"      pto.timg2col {left}, %image{suffix}{positions} "
            f"{{fmatrixMode = #pto.fmatrix_mode<a_auto>, config = #image_config}} : "
            f"{shape['left']}, {shape['image']}"
        )
        acc = "%acc, " if image else ""
        types = shape["acc"] + ", " if image else ""
        opname = "pto.tmatmul.acc" if image else "pto.tmatmul"
        lines.append(
            f"      {opname} ins({acc}{left}, %right : {types}{shape['left']}, {shape['right']}) "
            f"outs(%acc : {shape['acc']})"
        )
    output_offset = "%output_offset" if loop else "%c0"
    lines += [
        f"      pto.tmov ins(%acc : {shape['acc']}) outs(%result : {shape['result']}) "
        "{accToVecMode = #pto.acc_to_vec_mode<single_mode_vec0>}",
        f"      %op = pto.partition_view %ov, offsets = [{output_offset}, %c0], "
        f"sizes = [%c{p.rows}, %c16] : !pto.tensor_view<?x?xf32> -> "
        f"!pto.partition_tensor_view<{p.rows}x16xf32>",
        f"      pto.tstore ins(%result : {shape['result']}) "
        f"outs(%op : !pto.partition_tensor_view<{p.rows}x16xf32>)",
    ]
    if loop:
        lines.append("    }")
    config = f"#image_config = #pto.conv_tile_config<fmap_h={p.h}, fmap_w={p.w}, "
    config += (
        f"pad_list={p.pl}x{p.pr}x{p.pt}x{p.pb}, filter_h={p.kh}, filter_w={p.kw}, "
    )
    config += (
        f"dilation_h={p.dh}, dilation_w={p.dw}, stride_h={p.sh}, stride_w={p.sw}, "
    )
    config += f"pad_value=0 : i32, channel_size={p.c}, repeat_stride=0, repeat_time=1, "
    config += "repeat_mode=0, dst_stride=0, dst_mposition=0, transpose=false>"
    return (
        config
        + '\nmodule attributes {pto.target_arch = "a1"} {\n'
        + (
            "  func.func @a1_img2col_mat(%out: !pto.ptr<f32>, %input: !pto.ptr<f16>, "
            "%weight: !pto.ptr<f16>) attributes {pto.entry} {\n"
        )
        + "    pto.section.cube {\n"
        + "\n".join("  " + line for line in lines)
        + "\n    }\n    return\n  }\n}\n"
    )


def require_dependency(segment: str, source: str, destination: str) -> None:
    flags = re.finditer(
        rf"set_flag\(PIPE_{source}, PIPE_{destination}, (EVENT_ID[0-9]+)\);",
        segment,
    )
    for flag in flags:
        wait = f"wait_flag(PIPE_{source}, PIPE_{destination}, {flag.group(1)});"
        if wait in segment[flag.end() :]:
            return
    raise RuntimeError(f"Missing ordered {source}->{destination} set/wait pair")


def check_image_dependencies(code: str, p: Profile) -> None:
    writer = "TINSERT" if p.ub_stage else "TLOAD"
    pipe = "MTE3" if p.ub_stage else "MTE2"
    for im2col in re.finditer(r"TIMG2COL<[^;]+;", code):
        before = code[: im2col.start()]
        images = re.findall(r"__cbuf__ half\* \w+ = (\w+)\.data\(\);", before)
        if not images:
            raise RuntimeError("Missing original Mat image pointer")
        writes = list(re.finditer(rf"{writer}\({images[-1]},[^;]+;", before))
        if not writes:
            raise RuntimeError("Missing producer for the consumed image")
        require_dependency(code[writes[-1].end() : im2col.start()], pipe, "MTE1")
    if p.ub_stage:
        for insert in re.finditer(r"TINSERT\(\w+, (\w+),[^;]+;", code):
            loads = list(
                re.finditer(rf"TLOAD\({insert.group(1)},[^;]+;", code[: insert.start()])
            )
            if not loads:
                raise RuntimeError("Missing producer for UB staging bank")
            require_dependency(code[loads[-1].end() : insert.start()], "MTE2", "MTE3")


def check_reload_dependencies(code: str, p: Profile) -> None:
    if p.images <= 1 or p.sum_images:
        return
    body = code.split("for (", 1)[1].split("\n  }", 1)[0]
    im2col = body.index("TIMG2COL<")
    if not p.ub_stage:
        # The release is after consumption; its wait protects the next load.
        begin = body.index("TLOAD(")
        require_dependency(body[im2col:] + body[:begin], "MTE1", "MTE2")
        return
    # MTE3 is ordered: the output store completes the MTE1 -> M -> V ->
    # MTE3 chain before next iteration's TINSERT can overwrite the image.
    gemm = body.index("TMATMUL(", im2col)
    move = body.index("TMOV<", gemm)
    store = body.index("TSTORE(", move)
    require_dependency(body[im2col:gemm], "MTE1", "M")
    require_dependency(body[gemm:move], "M", "V")
    require_dependency(body[move:store], "V", "MTE3")


def check_structure(code: str, planned: str, p: Profile) -> None:
    if code.count("static AICORE inline void PTOAS__INIT_CONVTILE_CONFIG(") != 1:
        raise RuntimeError(
            "Expected exactly one shared image configuration initializer"
        )
    sites = 2 if p.sum_images else 1
    loads = sites * (p.c // 16 if p.ub_stage else 1)
    if code.count("TLOAD(") != loads + 1 or code.count("TIMG2COL<") != sites:
        raise RuntimeError("Unexpected image/weight DMA or im2col sites")
    if p.ub_stage and code.count("TINSERT(") != loads:
        raise RuntimeError("Missing C16 UB-to-L1 insertions")
    if "!pto.conv_tile_buf" in planned or "PTOAS__INIT_CONVTILE_CONFIG(" not in code:
        raise RuntimeError("Expected original Mat SSA and native-only image descriptor")
    producers = (
        (("MTE2", "MTE3"), ("MTE3", "MTE1")) if p.ub_stage else (("MTE2", "MTE1"),)
    )
    for source, destination in (*producers, ("MTE1", "M"), ("M", "V"), ("V", "MTE3")):
        if f"set_flag(PIPE_{source}, PIPE_{destination}," not in code:
            raise RuntimeError(f"Missing {source}->{destination} dependency")
    check_image_dependencies(code, p)
    check_reload_dependencies(code, p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "check"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=tuple(p.name for p in PROFILES))
    parser.add_argument("--level", choices=("level1", "level2", "level3"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.action == "generate":
        for p in PROFILES:
            for level in ("level1", "level2", "level3"):
                directory = output / f"{p.name}-{level}"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "input.pto").write_text(
                    make_program(p, explicit=level == "level3")
                )
                words = p.rows * 16 * (1 if p.sum_images else p.images) * 3
                print(f"{p.name}\t{level}\t{words}\t{' '.join(p.macros())}")
        return
    if args.profile is None or args.level is None:
        parser.error("check requires --profile and --level")
    p = next(p for p in PROFILES if p.name == args.profile)
    directory = output / f"{p.name}-{args.level}"
    check_structure(
        (directory / "generated.cpp").read_text(),
        (directory / "planned.pto").read_text(),
        p,
    )


if __name__ == "__main__":
    main()
