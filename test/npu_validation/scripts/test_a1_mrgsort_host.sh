#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# CPU oracle and native compilation only; never queries or opens a device.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ptoas_bin=$(command -v "${PTOAS:-ptoas}")
cpu_compiler=$(command -v "${CXX:-clang++-19}")
native_compiler=$(command -v "${CCEC:-ccec}")
python_bin=$(command -v "${PYTHON:-python3}")
: "${PTO_ISA_ROOT:?Set PTO_ISA_ROOT to the matching A1 merge ISA checkout}"
merge_work="${PTOAS_MRGSORT_TEST_OUTPUT_DIR:-$repo_root/build/a1-mrgsort-host}"
mkdir -p -- "$merge_work"
profiles=("32 32 0 0 8 false explicit level1" "32 32 0 0 64 true implicit level1"
          "64 96 0 0 160 false explicit level1" "64 96 0 0 160 true explicit level1"
          "32 64 96 0 128 false implicit level1" "32 64 96 0 192 true explicit level1"
          "64 96 128 160 448 false explicit level2" "64 96 128 160 448 true implicit level2"
          "512 512 0 0 600 false explicit level1" "512 512 0 0 600 true implicit level1"
          "64 96 128 0 288 true explicit level3" "32 32 32 32 128 false explicit level3")
pair_count=0
for settings in "${profiles[@]}"; do
    read -r col0 col1 col2 col3 out exhausted mode level <<< "$settings"
    profile="$merge_work/${settings// /-}"
    mkdir -p -- "$profile"
    "$python_bin" - "$repo_root/test/lit/a1/mrgsort.pto" "$settings" "$profile/input.pto" <<'PROFILE'
from pathlib import Path
import re
import sys

template, settings, output = sys.argv[1:]
c0, c1, c2, c3, out, exhausted, mode, level = settings.split()
columns, out = [int(c0), int(c1), int(c2), int(c3)], int(out)
ways = sum(c > 0 for c in columns)
scratch = 8 * sum(columns)
widths = {64: columns[0], 96: columns[1], 160: out, 1280: scratch}
text = re.sub(r'1x(64|96|160|1280)xf32', lambda match: f'1x{widths[int(match[1])]}xf32',
              Path(template).read_text())
for name, value in [('cols0', columns[0]), ('cols1', columns[1]), ('output_cols', out),
                    ('input_stride', max(columns) + 32), ('output_stride', out + 32)]:
    text = re.sub(rf'(%{name} = arith.constant )\d+( : index)', rf'\g<1>{value}\2', text)
extra = []
for source in range(2, ways):
    width = columns[source]
    extra += [f'    %cols{source} = arith.constant {width} : index',
              f'    %view{source} = pto.make_tensor_view %input{source}, shape = [%one, %input_stride], '
              'strides = [%input_stride, %one] : !pto.tensor_view<?x?xf32>',
              f'    %part{source} = pto.partition_view %view{source}, offsets = [%zero, %source_col], '
              f'sizes = [%one, %cols{source}] : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<1x{width}xf32>',
              f'    %source{source} = pto.alloc_tile : !pto.tile_buf<vec, 1x{width}xf32>',
              f'    pto.tload ins(%part{source} : !pto.partition_tensor_view<1x{width}xf32>) '
              f'outs(%source{source} : !pto.tile_buf<vec, 1x{width}xf32>)']
text = text.replace('    // EXTRA-SOURCES', '\n'.join(extra))
operands = ', '.join(f'%source{i}' for i in range(ways))
types = ', '.join(f'!pto.tile_buf<vec, 1x{c}xf32>' for c in columns[:ways])
if mode == 'explicit':
    operands += ', %scratch'
    types += f', !pto.tile_buf<vec, 1x{scratch}xf32>'
else:
    operands += ' no_tmp'
    text = '\n'.join(line for line in text.splitlines() if '%scratch = pto.alloc_tile' not in line) + '\n'
text = re.sub(r'pto.tmrgsort ins\([^\n]+?\) outs',
              f'pto.tmrgsort ins({operands} {{exhausted = {exhausted}}} : {types}) outs', text)
if level == 'level3':
    address = 0
    sizes = [(f'source{i}', c * 4) for i, c in enumerate(columns[:ways])]
    for name, size in sizes + [('result', out * 4), ('scratch', scratch * 4)]:
        text = text.replace(f'    %{name} = pto.alloc_tile :',
                            f'    %{name}_addr = arith.constant {address} : i64\n'
                            f'    %{name} = pto.alloc_tile addr = %{name}_addr :')
        address += size
    if address > 128 * 1024:
        raise ValueError('Test profile exceeds A1 UB capacity')
Path(output).write_text(text)
PROFILE
    "$ptoas_bin" --pto-arch=a1 "--pto-level=$level" "$profile/input.pto" -o "$profile/generated.cpp"
    grep -Fq 'TMRGSORT<' "$profile/generated.cpp"
    grep -Fq 'wait_flag(PIPE_MTE2, PIPE_V,' "$profile/generated.cpp"
    grep -Fq 'wait_flag(PIPE_MTE2, PIPE_S,' "$profile/generated.cpp"
    grep -Fq 'wait_flag(PIPE_S, PIPE_MTE3,' "$profile/generated.cpp"
    printf '#include <cstdint>\n' > "$profile/oracle.cpp"
    cat -- "$profile/generated.cpp" "$repo_root/test/npu_validation/scripts/a1_mrgsort_cpu.cpp" >> "$profile/oracle.cpp"
    ways=2
    if (( col2 > 0 )); then ways=3; fi
    if (( col3 > 0 )); then ways=4; fi
    "$cpu_compiler" -std=c++20 -O1 -D__CPU_SIM "-DMERGE_COLS0=$col0" "-DMERGE_COLS1=$col1" \
        "-DMERGE_COLS2=$col2" "-DMERGE_COLS3=$col3" "-DMERGE_WAYS=$ways" "-DMERGE_OUTPUT_COLS=$out" \
        "-DMERGE_EXHAUSTED=$exhausted" "-I$PTO_ISA_ROOT/include" "$profile/oracle.cpp" -o "$profile/oracle"
    "$profile/oracle"
    "$native_compiler" -c -O2 -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-m200 \
        "-I$PTO_ISA_ROOT/include" "$profile/generated.cpp" -o "$profile/native.o"
    pair_count=$((pair_count + out / 2 * 6))
done
printf 'A1 merge: %s CPU/offline profiles, %s pair slots bitwise; counts and guards checked; no device access.\n' \
    "${#profiles[@]}" "$pair_count"
