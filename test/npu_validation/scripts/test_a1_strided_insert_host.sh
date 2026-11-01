#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Driver-free lowering, bitwise payloads and A1 native-object checks.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ptoas_bin=$(command -v "${PTOAS:-ptoas}")
cxx_bin=$(command -v "${CXX:-clang++-19}")
native_bin=$(command -v "${CCEC:-ccec}")
python_bin=$(command -v "${PYTHON:-python3}")
readelf_bin=$(command -v readelf)
llvm_dis_bin=$(command -v llvm-dis-19 || command -v llvm-dis)
: "${PTO_ISA_ROOT:?Set PTO_ISA_ROOT to the matching row-step A1 ISA}"
rowstep_work="${PTOAS_ROWSTEP_OUTPUT_DIR:-$repo_root/build/a1-rowstep-host}"
helper="$repo_root/test/npu_validation/scripts/generate_a1_strided_insert_profiles.py"
mkdir -p -- "$rowstep_work"
"$python_bin" "$helper" generate --output-dir "$rowstep_work" > "$rowstep_work/profiles.tsv"
profile_count=0
output_values=0
while IFS=$'\t' read -r profile level values; do
    if [[ ! "$profile" =~ ^[0-9]{1,2}$ || ! "$values" =~ ^[0-9]{1,9}$ ]]; then
        printf 'Invalid row-step profile metadata\n' >&2
        exit 1
    fi
    case "$level" in
        level1|level2|level3) ;;
        *) printf 'Invalid row-step pipeline level\n' >&2; exit 1 ;;
    esac
    printf -v directory '%s/profile%02d-%s' "$rowstep_work" "$profile" "$level"
    "$ptoas_bin" --pto-arch=a1 --pto-level="$level" "$directory/input.pto" -o "$directory/generated.cpp"
    "$python_bin" "$helper" check --output-dir "$rowstep_work" --profile "$profile" --level "$level" \
        > "$directory/definitions.txt"
    mapfile -t definitions < "$directory/definitions.txt"
    for definition in "${definitions[@]}"; do
        if [[ ! "$definition" =~ ^-DPTO_TEST_ROWSTEP_[A-Z_]+=[0-9]+$ ]]; then
            printf 'Invalid row-step fixture definition\n' >&2
            exit 1
        fi
    done
    "$cxx_bin" -std=c++20 -O1 -D__CPU_SIM -D__DAV_CUBE__ -D__DAV_VEC__ \
        "-I$PTO_ISA_ROOT/include" "${definitions[@]}" "$directory/oracle.cpp" -o "$directory/oracle"
    PTO_CPU_SIM_ARCH=a2a3 PTO_CPU_SIM_L1_BYTES=1048576 PTO_CPU_SIM_UB_BYTES=131072 \
        PTO_CPU_SIM_NUM_CORES=1 PTO_CPU_SIM_TRACE_ENABLE=0 OMP_NUM_THREADS=1 "$directory/oracle"
    "$native_bin" -c -O2 -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-m200 \
        "-I$PTO_ISA_ROOT/include" "$directory/native.cpp" -o "$directory/native.o"
    "$native_bin" -c -emit-llvm -O2 -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-m200 \
        "-I$PTO_ISA_ROOT/include" "$directory/native.cpp" -o "$directory/native.bc"
    "$llvm_dis_bin" "$directory/native.bc" -o "$directory/native.ll"
    "$python_bin" "$helper" native --output-dir "$rowstep_work" --profile "$profile" --level "$level"
    "$readelf_bin" -rW "$directory/native.o" > "$directory/relocations.txt"
    if grep -Eq "Relocation section '\.rel(a)?\.text" "$directory/relocations.txt"; then
        printf 'Unexpected executable native text relocation\n' >&2
        exit 1
    fi
    profile_count=$((profile_count + 1))
    output_values=$((output_values + values))
    printf 'PASS profile%s %s: %s bitwise FP16 values, holes, guards and readonly GM\n' "$profile" "$level" "$values"
done < "$rowstep_work/profiles.tsv"
if [ "$profile_count" -ne 63 ]; then
    printf 'Expected all 21 profiles at all three pipeline levels\n' >&2
    exit 1
fi
printf 'Row-step insertion: %s CPU/native profiles, %s FP16 values; no device operations.\n' \
    "$profile_count" "$output_values"
