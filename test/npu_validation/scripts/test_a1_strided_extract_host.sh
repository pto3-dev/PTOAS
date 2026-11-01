#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# CPU and offline CCEC checks of unmodified compiler output; never opens a device.
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
ptoas_bin=$(command -v "${PTOAS:-ptoas}")
cxx_bin=$(command -v "${CXX:-clang++-19}")
native_bin=$(command -v "${CCEC:-ccec}")
python_bin=$(command -v "${PYTHON:-python3}")
readelf_bin=$(command -v readelf)
llvm_dis_bin=$(command -v llvm-dis-19 || command -v llvm-dis)
: "${PTO_ISA_ROOT:?Set PTO_ISA_ROOT to the matching block-step A1 ISA}"
extract_work="${PTOAS_EXTRACT_OUTPUT_DIR:-$repo_root/build/a1-blockstep-host}"
helper="$repo_root/test/npu_validation/scripts/generate_a1_strided_extract_profiles.py"
mkdir -p -- "$extract_work"
"$python_bin" "$helper" generate --output-dir "$extract_work" "$@" > "$extract_work/profiles.tsv"
expected_count=$(wc -l < "$extract_work/profiles.tsv")
if [ "$#" -eq 0 ] && [ "$expected_count" -ne 54 ]; then
    printf 'Expected all 18 profiles at all three pipeline levels\n' >&2
    exit 1
fi
profile_count=0
while IFS=$'\t' read -r profile level; do
    if [[ ! "$profile" =~ ^[0-9]{1,2}$ ]]; then
        printf 'Invalid extraction profile metadata\n' >&2
        exit 1
    fi
    case "$level" in
        level1|level2|level3) ;;
        *) printf 'Invalid extraction pipeline level\n' >&2; exit 1 ;;
    esac
    printf -v directory '%s/profile%02d-%s' "$extract_work" "$profile" "$level"
    "$ptoas_bin" --pto-arch=a1 --pto-level="$level" "$directory/input.pto" -o "$directory/generated.cpp"
    "$python_bin" "$helper" check --output-dir "$extract_work" --profile "$profile" --level "$level" \
        > "$directory/definitions.txt"
    mapfile -t definitions < "$directory/definitions.txt"
    for definition in "${definitions[@]}"; do
        if [[ ! "$definition" =~ ^-DPTO_TEST_EXTRACT_[A-Z_0-9]+=[0-9]+$ ]]; then
            printf 'Invalid extraction fixture definition\n' >&2
            exit 1
        fi
    done
    for variant in debug release; do
        cpu_flags=(-std=c++20 -O1 -D__CPU_SIM -D__DAV_CUBE__ -D__DAV_VEC__)
        if [ "$variant" = release ]; then
            cpu_flags+=(-DNDEBUG)
        fi
        "$cxx_bin" "${cpu_flags[@]}" "-I$PTO_ISA_ROOT/include" "${definitions[@]}" \
            "$directory/oracle.cpp" -o "$directory/oracle-$variant"
        PTO_CPU_SIM_ARCH=a2a3 PTO_CPU_SIM_L1_BYTES=1048576 PTO_CPU_SIM_UB_BYTES=131072 \
            PTO_CPU_SIM_NUM_CORES=1 PTO_CPU_SIM_TRACE_ENABLE=0 OMP_NUM_THREADS=1 \
            "$directory/oracle-$variant" > "$directory/cpu-$variant.txt"
        grep -Fq 'PASS exact outputs, resident source and readonly GM/guards' "$directory/cpu-$variant.txt"
    done
    "$native_bin" -c -O2 -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-m200 \
        "-I$PTO_ISA_ROOT/include" "$directory/native.cpp" -o "$directory/native.o"
    "$native_bin" -c -emit-llvm -O2 -x cce -std=c++17 --cce-aicore-only --cce-aicore-arch=dav-m200 \
        "-I$PTO_ISA_ROOT/include" "$directory/native.cpp" -o "$directory/native.bc"
    "$llvm_dis_bin" "$directory/native.bc" -o "$directory/native.ll"
    "$python_bin" "$helper" native --output-dir "$extract_work" --profile "$profile" --level "$level"
    "$readelf_bin" -rW "$directory/native.o" > "$directory/relocations.txt"
    if grep -Eq "Relocation section '\.rel(a)?\.text" "$directory/relocations.txt"; then
        printf 'Unexpected executable native text relocation\n' >&2
        exit 1
    fi
    profile_count=$((profile_count + 1))
    printf 'PASS profile%s %s: bitwise CPU, native object and Cube/DMA checks\n' "$profile" "$level"
done < "$extract_work/profiles.tsv"
if [ "$profile_count" -eq 0 ] || [ "$profile_count" -ne "$expected_count" ]; then
    printf 'Incomplete extraction profile run\n' >&2
    exit 1
fi
printf 'Block-step extraction: %s CPU/native profiles; no hardware execution.\n' "$profile_count"
