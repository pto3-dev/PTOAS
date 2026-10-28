# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile emitted A1 multi-K GEMM through CPU and native public PTO-ISA."""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
simpler = pytest.importorskip("simpler_setup")


@pytest.fixture(scope="module", autouse=True)
def require_a1():
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    acl.aclrtGetSocName.argtypes = []
    soc = acl.aclrtGetSocName()
    if not soc or not soc.startswith(b"Ascend310P"):
        pytest.skip("Requires Ascend310P hardware")


@pytest.mark.parametrize(
    "rows,zz,ub_input",
    [(32, False, False), (224, True, False), (32, False, True), (224, True, True)],
)
def test_a1_multi_k_emitted_cpp(tmp_path, rows, zz, ub_input):
    root = Path(__file__).resolve().parents[3]
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa or not (Path(isa) / "include/pto/npu/a1/TMatrix.hpp").is_file():
        pytest.skip("PTO_ISA_ROOT must point to public A1 multi-K headers")
    ptoas = root / "build/tools/ptoas/ptoas"
    if not ptoas.is_file():
        pytest.skip("Build this PTOAS worktree first")
    compiler = (
        shutil.which("clang++-19") or shutil.which("g++-15") or shutil.which("g++-14")
    )
    if compiler is None:
        pytest.skip("CPU-first oracle requires Clang19 or GCC>=14")
    name = "vec_to_mat_gemm.pto" if ub_input else "multi_k_gemm.pto"
    source = (root / "test/lit/a1" / name).read_text()
    source = source.replace("a1_vec_to_mat_gemm", "a1_multi_k_gemm")
    factor = 4 if ub_input else 2
    source = source.replace(
        "arith.constant 32 : index", f"arith.constant {rows} : index"
    )
    source = source.replace("32x", f"{rows}x")
    if zz:
        source = source.replace(
            f"mat, {rows}x144xf16, blayout=col_major",
            f"mat, {rows}x144xf16, blayout=row_major",
        )
    input_ir = tmp_path / "gemm.pto"
    output_cpp = tmp_path / "gemm.cpp"
    input_ir.write_text(source)
    subprocess.run(
        [str(ptoas), "--pto-arch=a1", str(input_ir), "-o", str(output_cpp)],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = output_cpp.read_text()
    assert "TileType::Left, half," in generated and "TMATMUL_ACC(" in generated
    assert "PIPE_FIX" not in generated
    if ub_input:
        assert "set_flag(PIPE_V, PIPE_MTE3," in generated
        assert "set_flag(PIPE_MTE3, PIPE_MTE1," in generated
    cpu_main = r"""
#include <vector>
int main() {
    std::vector<half> a(M*144), b(144*16);
    std::vector<float> output(M*16+32,7);
    for(int i=0;i<M*144;++i) a[i]=half((i%17-8)/16.0f);
    for(int i=0;i<144*16;++i) b[i]=half((i%13-6)/16.0f);
    a1_multi_k_gemm(output.data()+16,a.data(),b.data());
    for(int row=0;row<M;++row) for(int col=0;col<16;++col) {
        float expected=0;
        for(int k=0;k<144;++k) expected+=float(a[row*144+k])*float(b[k*16+col]);
        if(output[16+row*16+col]!=2*expected) return 1;
    }
    for(int i=0;i<16;++i) if(output[i]!=7 || output[16+M*16+i]!=7) return 2;
    return 0;
}
""".replace("M", str(rows)).replace("2*expected", f"{factor}*expected")
    cpu = tmp_path / "cpu.cpp"
    cpu.write_text("#include <cstdint>\n" + generated + cpu_main)
    executable = tmp_path / "cpu-gemm"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            "-D__DAV_VEC__",
            f"-I{Path(isa) / 'include'}",
            str(cpu),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativeCompilerGemm(__gm__ int64_t* args) {
    auto* a=reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* b=reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto* out=reinterpret_cast<__gm__ Tensor*>(args[2]);
    a1_multi_k_gemm(reinterpret_cast<__gm__ float*>(out->buffer.addr)+16,
                   reinterpret_cast<__gm__ half*>(a->buffer.addr)+16,
                   reinterpret_cast<__gm__ half*>(b->buffer.addr)+16);
}
"""
    native = tmp_path / "native.cpp"
    # Adapt the top-level ABI only; execute the generated body unchanged.
    native_generated = generated.replace(
        'extern "C" __global__ AICORE void a1_multi_k_gemm',
        "AICORE void a1_multi_k_gemm",
    )
    assert native_generated != generated
    native.write_text("#include <cstdint>\n" + native_generated + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(native), isa)
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in (907, 911):
            generator = torch.Generator().manual_seed(seed)
            a = (
                torch.randint(-8, 9, (rows, 144), generator=generator).float() / 16
            ).half()
            b = (
                torch.randint(-8, 9, (144, 16), generator=generator).float() / 16
            ).half()
            ga = torch.full((a.numel() + 32,), 7, dtype=torch.float16)
            ga[16:-16] = a.flatten()
            gb = torch.full((b.numel() + 32,), 7, dtype=torch.float16)
            gb[16:-16] = b.flatten()
            output = torch.full((rows * 16 + 32,), 7, dtype=torch.float32)
            runner(ga, gb, output)
            assert torch.equal(
                output[16:-16].reshape(rows, 16), factor * (a.float() @ b.float())
            )
            assert torch.all(output[:16] == 7) and torch.all(output[-16:] == 7)
            assert torch.equal(ga[16:-16], a.flatten()) and torch.equal(
                gb[16:-16], b.flatten()
            )
            assert torch.all(ga[:16] == 7) and torch.all(ga[-16:] == 7)
            assert torch.all(gb[:16] == 7) and torch.all(gb[-16:] == 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
