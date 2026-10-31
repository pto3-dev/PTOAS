#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Driver-free zero-extension -> bounded GM-row selection with raw-bit guards.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ptoas_bin=$(command -v "${PTOAS:-ptoas}")
cxx_bin=$(command -v "${CXX:-clang++-19}")
native_bin=$(command -v "${CCEC:-ccec}")
python_bin=$(command -v "${PYTHON:-python3}")
: "${PTO_ISA_ROOT:?Set PTO_ISA_ROOT to the matching A1 ISA}"
bounded_work="${PTOAS_BOUND_TEST_OUTPUT_DIR:-$repo_root/build/a1-bounded-integer-dma-host}"
mkdir -p -- "$bounded_work"
while read -r mode rows columns remainder; do
    oracle_mode=0
    if [ "$mode" = narrow8 ]; then oracle_mode=1; fi
    if [ "$mode" = narrow16 ]; then oracle_mode=2; fi
    for level in level1 level2; do
        profile="$bounded_work/$mode-$rows-$remainder-$level"
        mkdir -p -- "$profile"
        "$python_bin" - "$repo_root/test/lit/a1/bounded_integer_dma.pto" "$profile/input.pto" \
            "$mode" "$rows" "$columns" "$remainder" <<'PYCODE'
from pathlib import Path
import sys
source, output, mode, rows, columns, remainder = sys.argv[1:]
text = Path(source).read_text().replace("constant 336 : index", f"constant {rows} : index")
text = text.replace("constant 96 : index", f"constant {columns} : index")
text = text.replace("1x96xf32", f"1x{columns}xf32")
cast = "%index = arith.index_castui %word : i32 to index"
if mode == "extend32":
    replacement = "%wide = arith.extui %word : i32 to i64\n      %index = arith.index_cast %wide : i64 to index"
    text = text.replace(cast, replacement)
elif mode in ("narrow8", "narrow16"):
    bits = mode.removeprefix("narrow")
    replacement = f"%narrow = arith.trunci %word : i32 to i{bits}\n      "
    replacement += f"%index = arith.index_castui %narrow : i{bits} to index"
    text = text.replace(cast, replacement)
if remainder == "unsigned":
    text = text.replace("arith.remsi", "arith.remui")
elif remainder == "none":
    text = text.replace("      %selected_row = arith.remsi %index, %rows : index\n", "")
    text = text.replace("%selected_row", "%index")
Path(output).write_text(text)
PYCODE
        "$ptoas_bin" --pto-arch=a1 --pto-level="$level" "$profile/input.pto" -o "$profile/generated.cpp"
        "$python_bin" - "$profile/generated.cpp" <<'PYCODE'
from pathlib import Path
import sys
code = Path(sys.argv[1]).read_text()
assert code.count("TLOAD(") == 2 and code.count("TSTORE(") == 1
assert code.count(".GetValue(") == 1
assert "set_flag(PIPE_MTE2, PIPE_S," in code
assert "wait_flag(PIPE_MTE2, PIPE_S," in code
PYCODE
        printf '#include <cstdint>\n' > "$profile/oracle.cpp"
        cat -- "$profile/generated.cpp" \
            "$repo_root/test/npu_validation/scripts/a1_bounded_integer_dma_cpu.cpp" >> "$profile/oracle.cpp"
        "$cxx_bin" -std=c++20 -O1 -D__CPU_SIM "-DBOUNDED_ROWS=$rows" "-DBOUNDED_COLS=$columns" \
            "-DBOUNDED_MODE=$oracle_mode" "-I$PTO_ISA_ROOT/include" "$profile/oracle.cpp" -o "$profile/oracle"
        "$profile/oracle"
        "$native_bin" -c -O2 -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-m200 \
            "-I$PTO_ISA_ROOT/include" "$profile/generated.cpp" -o "$profile/native.o"
    done
done <<'PROFILES'
index32 336 96 signed
index32 8400 96 signed
index32 336 96 unsigned
index32 8400 96 unsigned
extend32 336 96 signed
extend32 8400 96 signed
narrow8 256 8 none
narrow16 65536 8 none
PROFILES
printf 'A1 bounded integer DMA: 16 CPU/native profiles, 56832 raw output words and guards passed.\n'
