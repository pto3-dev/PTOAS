#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Driver-free bitwise, buffer-lifetime and dav-m200 compilation checks.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ptoas_bin=$(command -v "${PTOAS:-ptoas}")
cxx_bin=$(command -v "${CXX:-clang++-19}")
native_bin=$(command -v "${CCEC:-ccec}")
python_bin=$(command -v "${PYTHON:-python3}")
: "${PTO_ISA_ROOT:?Set PTO_ISA_ROOT to the matching A1 ISA}"
mat_work="${PTOAS_IMG2COL_MAT_OUTPUT_DIR:-$repo_root/build/a1-img2col-mat-host}"
helper="$repo_root/test/npu_validation/scripts/generate_a1_img2col_mat_profiles.py"
mkdir -p -- "$mat_work"
"$python_bin" "$helper" generate --output-dir "$mat_work" > "$mat_work/profiles.tsv"
profile_count=0
output_words=0
while IFS=$'\t' read -r profile level words definitions; do
    case "$level" in
        level1|level2|level3) ;;
        *) printf 'Invalid fixture pipeline level\n' >&2; exit 1 ;;
    esac
    if [[ ! "$words" =~ ^[0-9]{1,9}$ ]]; then
        printf 'Invalid fixture output count\n' >&2
        exit 1
    fi
    read -r -a mat_definitions <<< "$definitions"
    for definition in "${mat_definitions[@]}"; do
        if [[ ! "$definition" =~ ^-DIMG_[A-Z0-9_]+=[0-9]+$ ]]; then
            printf 'Invalid fixture definition\n' >&2
            exit 1
        fi
    done
    directory="$mat_work/$profile-$level"
    "$ptoas_bin" --pto-arch=a1 --pto-level="$level" "$directory/input.pto" -o "$directory/generated.cpp"
    "$ptoas_bin" --pto-arch=a1 --pto-level="$level" --emit-pto-ir "$directory/input.pto" -o "$directory/planned.pto"
    "$python_bin" "$helper" check --output-dir "$mat_work" --profile "$profile" --level "$level"
    printf '#include <cstdint>\n' > "$directory/oracle.cpp"
    cat -- "$directory/generated.cpp" \
        "$repo_root/test/npu_validation/scripts/a1_img2col_mat_cpu.cpp" >> "$directory/oracle.cpp"
    "$cxx_bin" -std=c++20 -O1 -ffp-contract=off -D__CPU_SIM -D__DAV_CUBE__ -D__DAV_VEC__ \
        "-I$PTO_ISA_ROOT/include" "${mat_definitions[@]}" "$directory/oracle.cpp" -o "$directory/oracle"
    "$directory/oracle"
    "$native_bin" -c -O2 -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-m200 \
        "-I$PTO_ISA_ROOT/include" "$directory/generated.cpp" -o "$directory/native.o"
    profile_count=$((profile_count + 1))
    output_words=$((output_words + words))
    printf 'PASS %s %s: %s bitwise FP32 values, readonly inputs and guards\n' "$profile" "$level" "$words"
done < "$mat_work/profiles.tsv"
printf 'Mat im2col: %s CPU/native profiles, %s FP32 values; no device queries or execution.\n' \
    "$profile_count" "$output_words"
