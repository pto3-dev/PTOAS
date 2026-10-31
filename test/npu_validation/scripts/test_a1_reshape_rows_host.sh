#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Driver-free raw-bit alias oracle and native compile; never queries or opens an NPU.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ptoas_bin=$(command -v "${PTOAS:-ptoas}")
cxx_bin=$(command -v "${CXX:-clang++-19}")
native_bin=$(command -v "${CCEC:-ccec}")
: "${PTO_ISA_ROOT:?Set PTO_ISA_ROOT to the matching A1 reshape ISA}"
rows_work="${PTOAS_ROW_VIEW_TEST_OUTPUT_DIR:-$repo_root/build/a1-reshape-rows-host}"
mkdir -p -- "$rows_work"
for dtype in f32 f16; do
    ctype=float
    bits=uint32_t
    if [ "$dtype" = f16 ]; then ctype=half; bits=uint16_t; fi
    for level in level1 level2; do
        profile="$rows_work/$dtype-$level"
        mkdir -p -- "$profile"
        sed "s/f32/$dtype/g" "$repo_root/test/lit/a1/reshape_row_subview.pto" > "$profile/input.pto"
        "$ptoas_bin" --pto-arch=a1 --pto-level="$level" "$profile/input.pto" -o "$profile/generated.cpp"
        grep -Fq 'TRESHAPE(' "$profile/generated.cpp"
        grep -Fq 'wait_flag(PIPE_MTE3, PIPE_MTE2,' "$profile/generated.cpp"
        printf '#include <cstdint>\n' > "$profile/oracle.cpp"
        cat -- "$profile/generated.cpp" \
            "$repo_root/test/npu_validation/scripts/a1_reshape_rows_cpu.cpp" >> "$profile/oracle.cpp"
        "$cxx_bin" -std=c++20 -O1 -D__CPU_SIM "-DROW_VIEW_CTYPE=$ctype" "-DROW_VIEW_BITS=$bits" \
            "-I$PTO_ISA_ROOT/include" "$profile/oracle.cpp" -o "$profile/oracle"
        "$profile/oracle"
        "$native_bin" -c -O2 -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-m200 \
            "-I$PTO_ISA_ROOT/include" "$profile/generated.cpp" -o "$profile/native.o"
    done
done
printf 'A1 reshape rows: four CPU/native-offline profiles, 7680 output words bitwise and guards passed.\n'
