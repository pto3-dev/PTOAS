#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Driver-free typed scalar reads after DMA and vector producers; no NPU calls.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ptoas_bin=$(command -v "${PTOAS:-ptoas}")
cxx_bin=$(command -v "${CXX:-clang++-19}")
native_bin=$(command -v "${CCEC:-ccec}")
: "${PTO_ISA_ROOT:?Set PTO_ISA_ROOT to the matching A1 ISA}"
read_work="${PTOAS_READ_TEST_OUTPUT_DIR:-$repo_root/build/a1-integer-scalar-read-host}"
mkdir -p -- "$read_work"
for dtype in i32 si32 ui32; do
    ctype=int32_t
    if [ "$dtype" = ui32 ]; then ctype=uint32_t; fi
    for level in level1 level2; do
        profile="$read_work/$dtype-$level"
        mkdir -p -- "$profile"
        python3 - "$repo_root/test/lit/a1/integer_scalar_read.pto" "$profile/input.pto" "$dtype" <<'PYCODE'
from pathlib import Path
import sys
source, output, dtype = sys.argv[1:]
text = Path(source).read_text().replace("i32", dtype)
if dtype != "i32":
    # arith.bitcast accepts signless integers; this bridge keeps all 32 bits.
    text = text.replace(
        f"%bits = arith.bitcast %word : {dtype} to f32",
        f"%signless = builtin.unrealized_conversion_cast %word : {dtype} to i32\n"
        "      %bits = arith.bitcast %signless : i32 to f32",
    )
Path(output).write_text(text)
PYCODE
        "$ptoas_bin" --pto-arch=a1 --pto-level="$level" "$profile/input.pto" -o "$profile/generated.cpp"
        python3 - "$profile/generated.cpp" "$ctype" <<'PYCODE'
from pathlib import Path
import re
import sys
code = Path(sys.argv[1]).read_text()
assert code.count("TLOAD(") == code.count("TSTORE(") == code.count("TGATHER<") == 1
assert code.count(".GetValue(") == code.count(".SetValue(") == 2
assert len(re.findall(r"\b" + sys.argv[2] + r" \w+ = .*\.GetValue\(", code)) == 2
for source, destination in (("MTE2", "S"), ("V", "S"), ("S", "MTE3")):
    assert f"set_flag(PIPE_{source}, PIPE_{destination}," in code
    assert f"wait_flag(PIPE_{source}, PIPE_{destination}," in code
PYCODE
        printf '#include <cstdint>\n' > "$profile/oracle.cpp"
        cat -- "$profile/generated.cpp" \
            "$repo_root/test/npu_validation/scripts/a1_integer_scalar_read_cpu.cpp" >> "$profile/oracle.cpp"
        "$cxx_bin" -std=c++20 -O1 -D__CPU_SIM "-DREAD_TYPE=$ctype" \
            "-I$PTO_ISA_ROOT/include" "$profile/oracle.cpp" -o "$profile/oracle"
        "$profile/oracle"
        "$native_bin" -c -O2 -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-m200 \
            "-I$PTO_ISA_ROOT/include" "$profile/generated.cpp" -o "$profile/native.o"
    done
done
printf 'A1 integer scalar reads: 6 generated CPU/native profiles, 4608 raw output words and guards passed.\n'
