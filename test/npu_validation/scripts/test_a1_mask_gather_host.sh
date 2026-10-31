#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Driver-free generated-code bitwise oracle and dav-m200 compilation only.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ptoas_bin=$(command -v "${PTOAS:-ptoas}")
cxx_bin=$(command -v "${CXX:-clang++-19}")
native_bin=$(command -v "${CCEC:-ccec}")
: "${PTO_ISA_ROOT:?Set PTO_ISA_ROOT to the matching A1 mask gather ISA}"
mask_work="${PTOAS_MASK_TEST_OUTPUT_DIR:-$repo_root/build/a1-mask-gather-host}"
mkdir -p -- "$mask_work"
for dtype in f32 f16 i32 ui32; do
    for pair in even-odd outer-quarter inner-quarter identity; do
        case "$pair" in
            even-odd) mask_a=P0101; mask_b=P1010; factor=2; offset_a=0; offset_b=1 ;;
            outer-quarter) mask_a=P0001; mask_b=P1000; factor=4; offset_a=0; offset_b=3 ;;
            inner-quarter) mask_a=P0010; mask_b=P0100; factor=4; offset_a=1; offset_b=2 ;;
            identity) mask_a=P1111; mask_b=P1111; factor=1; offset_a=0; offset_b=0 ;;
        esac
        columns=608
        definitions=()
        if [ "$dtype" = f16 ]; then
            definitions+=(-DMASK_FP16)
            if [ "$factor" = 4 ]; then columns=704; fi
        elif [ "$dtype" = i32 ]; then
            definitions+=(-DMASK_CTYPE=int32_t)
        elif [ "$dtype" = ui32 ]; then
            definitions+=(-DMASK_CTYPE=uint32_t)
        fi
        definitions+=("-DMASK_SOURCE_COLS=$columns" "-DMASK_FACTOR=$factor"
                      "-DMASK_OFFSET_A=$offset_a" "-DMASK_OFFSET_B=$offset_b")
        for level in level1 level2; do
            profile="$mask_work/$dtype-$pair-$level"
            mkdir -p -- "$profile"
            python3 - "$repo_root/test/lit/a1/gather_mask.pto" "$profile/input.pto" \
                "$dtype" "$columns" "$factor" "$mask_a" "$mask_b" <<'PYCODE'
from pathlib import Path
import sys
source, output, dtype, columns, factor, mask_a, mask_b = sys.argv[1:]
text = Path(source).read_text().replace("608", columns).replace("304", str(int(columns) // int(factor)))
text = text.replace("P0101", mask_a).replace("P1010", mask_b)
if dtype == "f16":
    text = text.replace("ui32", "f32").replace("f32", "f16")
elif dtype in ("i32", "ui32"):
    text = text.replace("f32", dtype)
Path(output).write_text(text)
PYCODE
            "$ptoas_bin" --pto-arch=a1 --pto-level="$level" "$profile/input.pto" -o "$profile/generated.cpp"
            python3 - "$profile/generated.cpp" <<'PYCODE'
from pathlib import Path
import sys
code = Path(sys.argv[1]).read_text()
assert code.count("TLOAD(") == 1
assert code.count("TGATHER<") == 2
assert code.count("TSTORE(") == 2
assert code.count("TASSIGN(") == 3, "mask gather must not allocate indices or scratch"
assert "wait_flag(PIPE_MTE2, PIPE_V," in code
assert "wait_flag(PIPE_V, PIPE_MTE3," in code
assert "PIPE_MTE2, PIPE_S," not in code
PYCODE
            printf '#include <cstdint>\n' > "$profile/oracle.cpp"
            cat -- "$profile/generated.cpp" \
                "$repo_root/test/npu_validation/scripts/a1_mask_gather_cpu.cpp" >> "$profile/oracle.cpp"
            "$cxx_bin" -std=c++20 -O1 -D__CPU_SIM "${definitions[@]}" \
                "-I$PTO_ISA_ROOT/include" "$profile/oracle.cpp" -o "$profile/oracle"
            "$profile/oracle"
            "$native_bin" -c -O2 -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-m200 \
                "-I$PTO_ISA_ROOT/include" "$profile/generated.cpp" -o "$profile/native.o"
        done
    done
done
printf 'A1 mask gather: 32 generated CPU/native-offline profiles, 117888 output words bitwise and guards passed.\n'
