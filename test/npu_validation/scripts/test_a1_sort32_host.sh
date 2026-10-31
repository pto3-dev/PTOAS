#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# CPU raw-bit oracle and native compilation only: never queries or opens an NPU.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ptoas_bin=$(command -v "${PTOAS:-ptoas}")
cpu_compiler=$(command -v "${CXX:-clang++-19}")
native_compiler=$(command -v "${CCEC:-ccec}")
python_bin=$(command -v "${PYTHON:-python3}")
: "${PTO_ISA_ROOT:?Set PTO_ISA_ROOT to the matching A1 sort32 ISA checkout}"
sort_work="${PTOAS_SORT32_TEST_OUTPUT_DIR:-$repo_root/build/a1-sort32-host}"
mkdir -p -- "$sort_work"
profiles=("1 32 explicit level1" "1 32 implicit level1" "8 96 explicit level1" "8 96 implicit level1"
          "2 320 explicit level1" "2 320 implicit level1" "1 1024 explicit level1" "1 1024 implicit level1"
          "3 96 explicit level2" "3 96 implicit level2" "3 96 explicit level3")
pair_count=0
for settings in "${profiles[@]}"; do
    read -r rows columns mode level <<< "$settings"
    profile="$sort_work/$rows-$columns-$mode-$level"
    mkdir -p -- "$profile"
    "$python_bin" - "$repo_root/test/lit/a1/sort32.pto" "$rows" "$columns" "$mode" "$level" \
        "$profile/input.pto" <<'PROFILE'
from pathlib import Path
import sys

template, rows, columns, mode, level, output = sys.argv[1:]
rows, columns = int(rows), int(columns)
text = Path(template).read_text().replace("8x192x", f"{rows}x{2 * columns}x").replace("8x96x", f"{rows}x{columns}x")
for name, old, new in [("rows", 8, rows), ("cols", 96, columns), ("output_cols", 192, 2 * columns),
                       ("input_stride", 112, columns + 16), ("output_stride", 224, 2 * columns + 32)]:
    text = text.replace(f"%{name} = arith.constant {old} : index", f"%{name} = arith.constant {new} : index")
if mode == "implicit":
    text = "\n".join(line for line in text.splitlines() if "%scratch = pto.alloc_tile" not in line) + "\n"
    text = text.replace("%source, %indices, %scratch :", "%source, %indices :")
    text = text.replace(", !pto.tile_buf<vec, 1x512xf32>) outs", ") outs")
if level == "level3":
    address = 0
    for name, size in [("source", rows * columns * 4), ("indices", rows * columns * 4),
                       ("result", rows * columns * 8), ("scratch", 2048)]:
        text = text.replace(f"    %{name} = pto.alloc_tile :",
                            f"    %{name}_addr = arith.constant {address} : i64\n"
                            f"    %{name} = pto.alloc_tile addr = %{name}_addr :")
        address += size
    if address > 128 * 1024:
        raise ValueError("Test profile exceeds A1 UB capacity")
Path(output).write_text(text)
PROFILE
    "$ptoas_bin" --pto-arch=a1 "--pto-level=$level" "$profile/input.pto" -o "$profile/generated.cpp"
    grep -Fq 'TSORT32(' "$profile/generated.cpp"
    grep -Fq 'wait_flag(PIPE_MTE2, PIPE_V,' "$profile/generated.cpp"
    grep -Fq 'wait_flag(PIPE_S, PIPE_MTE3,' "$profile/generated.cpp"
    printf '#include <cstdint>\n' > "$profile/oracle.cpp"
    cat -- "$profile/generated.cpp" "$repo_root/test/npu_validation/scripts/a1_sort32_cpu.cpp" >> "$profile/oracle.cpp"
    "$cpu_compiler" -std=c++20 -O1 -D__CPU_SIM "-DSORT_ROWS=$rows" "-DSORT_COLS=$columns" \
        "-I$PTO_ISA_ROOT/include" "$profile/oracle.cpp" -o "$profile/oracle"
    "$profile/oracle"
    "$native_compiler" -c -O2 -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-m200 \
        "-I$PTO_ISA_ROOT/include" "$profile/generated.cpp" -o "$profile/native.o"
    pair_count=$((pair_count + rows * columns * 3))
done
# Generic CPU TSORT32 returns without writing for broadcast indices. Keep the
# original generated code; cover this profile offline and in ISA native-model tests.
"$ptoas_bin" --pto-arch=a1 "$repo_root/test/lit/a1/sort32_broadcast.pto" -o "$sort_work/broadcast.cpp"
"$native_compiler" -c -O2 -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-m200 \
    "-I$PTO_ISA_ROOT/include" "$sort_work/broadcast.cpp" -o "$sort_work/broadcast.o"
printf 'A1 sort32: %s CPU/offline profiles, %s pairs bitwise; one broadcast offline profile; no device access.\n' \
    "${#profiles[@]}" "$pair_count"
