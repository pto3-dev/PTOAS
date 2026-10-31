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
: "${PTO_ISA_ROOT:?Set PTO_ISA_ROOT to the matching A1 indexed-gather ISA checkout}"
# All artifacts remain below the caller's managed build directory. Preserve
# them for diagnosis; do not allocate a predictable directory in shared /tmp.
gather_work="${PTOAS_GATHER_TEST_OUTPUT_DIR:-$repo_root/build/a1-indexed-gather-host}"
mkdir -p -- "$gather_work"
profile_count=0
for dtype in f16 f32 i32 ui32; do
    case "$dtype" in
        f16) data_type=half ;;
        f32) data_type=float ;;
        i32) data_type=int32_t ;;
        ui32) data_type=uint32_t ;;
    esac
    for index_dtype in i32 ui32; do
        case "$index_dtype" in
            i32) index_type=int32_t ;;
            ui32) index_type=uint32_t ;;
        esac
        for mode in explicit implicit; do
            profile="$gather_work/$dtype-$index_dtype-$mode"
            mkdir -p -- "$profile"
            if [ "$mode" = implicit ]; then
                sed -e '/%scratch = pto.alloc_tile/d' \
                    -e 's/%source, %indices, %scratch :/%source, %indices :/' \
                    -e 's/, !pto.tile_buf<vec, 8x192xi32>) outs/) outs/' \
                    "$repo_root/test/lit/a1/gather_index.pto" > "$profile/template.pto"
            else
                cp -- "$repo_root/test/lit/a1/gather_index.pto" "$profile/template.pto"
            fi
            # Specialize indices before values so a signed i32 source remains
            # independent from an unsigned index operand.
            sed -e "s/i32/$index_dtype/g" -e "s/f32/$dtype/g" \
                "$profile/template.pto" > "$profile/input.pto"
            "$ptoas_bin" --pto-arch=a1 "$profile/input.pto" -o "$profile/generated.cpp"
            grep -Fq 'TGATHER(' "$profile/generated.cpp"
            grep -Fq 'set_flag(PIPE_MTE2, PIPE_S,' "$profile/generated.cpp"
            grep -Fq 'wait_flag(PIPE_MTE2, PIPE_S,' "$profile/generated.cpp"
            grep -Fq 'set_flag(PIPE_MTE2, PIPE_V,' "$profile/generated.cpp"
            grep -Fq 'wait_flag(PIPE_V, PIPE_MTE3,' "$profile/generated.cpp"
            printf '#include <cstdint>\n' > "$profile/oracle.cpp"
            cat -- "$profile/generated.cpp" "$repo_root/test/npu_validation/scripts/a1_indexed_gather_cpu.cpp" \
                >> "$profile/oracle.cpp"
            "$cpu_compiler" -std=c++20 -O1 -D__CPU_SIM \
                "-DGATHER_DATA=$data_type" "-DGATHER_INDEX=$index_type" \
                "-I$PTO_ISA_ROOT/include" "$profile/oracle.cpp" -o "$profile/oracle"
            "$profile/oracle"
            "$native_compiler" -c -O2 -x cce -std=c++17 --cce-aicore-only \
                --cce-aicore-arch=dav-m200 "-I$PTO_ISA_ROOT/include" \
                "$profile/generated.cpp" -o "$profile/native.o"
            profile_count=$((profile_count + 1))
        done
    done
done
[ "$profile_count" -eq 16 ]
printf 'A1 gather: %s profiles, 49152 logical values bitwise, CPU/offline only.\n' "$profile_count"
