# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU-first and real 310P verification of N32 GEMM and accumulator subviews."""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
simpler = pytest.importorskip("simpler_setup")


@pytest.mark.parametrize("mode", ["single", "accumulate", "reuse"])
def test_a1_n32_gemm_subview(tmp_path, mode):
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    acl.aclrtGetSocName.argtypes = []
    soc = acl.aclrtGetSocName()
    if not soc or not soc.startswith(b"Ascend310P"):
        pytest.skip("Requires Ascend310P")
    root = Path(__file__).resolve().parents[3]
    isa = os.environ.get("PTO_ISA_ROOT")
    assert isa, "PTO_ISA_ROOT must select the A1 N32 ISA branch"
    compiler = shutil.which("clang++-19")
    assert compiler, "CPU oracle requires Clang19"
    source = (root / "test/lit/a1/n32_gemm_subview.pto").read_text()
    if mode != "single":
        operation = """    pto.tmatmul.acc ins(%acc, %left, %right : !pto.tile_buf<acc, 32x32xf32, blayout=col_major, slayout=row_major, fractal=1024>, !pto.tile_buf<left, 32x64xf16, blayout=row_major, slayout=row_major, fractal=512>, !pto.tile_buf<right, 64x32xf16, blayout=row_major, slayout=col_major, fractal=512>) outs(%acc : !pto.tile_buf<acc, 32x32xf32, blayout=col_major, slayout=row_major, fractal=1024>)\n"""
        anchor = (
            "    %lo = pto.subview"
            if mode == "accumulate"
            else "    pto.tstore ins(%v0"
        )
        assert source.count(anchor) == 1
        source = source.replace(anchor, operation + anchor)
    input_ir = tmp_path / "n32.pto"
    input_ir.write_text(source)
    cpp = tmp_path / "generated.cpp"
    subprocess.run(
        [
            str(root / "build/tools/ptoas/ptoas"),
            "--pto-arch=a1",
            "--pto-level=level3",
            str(input_ir),
            "-o",
            str(cpp),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = cpp.read_text()
    assert "TMATMUL(" in generated and generated.count("TMOV<") == 2
    assert "set_flag(PIPE_M, PIPE_V," in generated
    if mode != "single":
        assert "TMATMUL_ACC(" in generated
    if mode == "reuse":
        after_view_reads = generated.rsplit("TMOV<", 1)[1].split("TMATMUL_ACC(", 1)[0]
        assert "set_flag(PIPE_V, PIPE_M," in after_view_reads
        assert "wait_flag(PIPE_V, PIPE_M," in after_view_reads
    cpu_main = r"""
#include <vector>
int main() {
 std::vector<half> a(32*64), b(64*32); std::vector<float> out(32*32+32,7);
 for(int i=0;i<32*64;++i)a[i]=half((i%17-8)/16.0f);
 for(int i=0;i<64*32;++i)b[i]=half((i%13-6)/16.0f);
 a1_n32_gemm_subview(a.data(),b.data(),out.data()+16);
 for(int i=0;i<32;++i)for(int j=0;j<32;++j){float v=0;for(int k=0;k<64;++k)v+=float(a[i*64+k])*float(b[k*32+j]);if(out[16+i*32+j]!=FACTOR*v)return 1;}
 for(int i=0;i<16;++i)if(out[i]!=7||out[16+1024+i]!=7)return 2;
 return 0;
}
""".replace("FACTOR", "2" if mode == "accumulate" else "1")
    cpu = tmp_path / "cpu.cpp"
    cpu.write_text("#include <cstdint>\n" + generated + cpu_main)
    executable = tmp_path / "cpu"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O1",
            "-D__CPU_SIM",
            "-D__DAV_CUBE__",
            "-D__DAV_VEC__",
            f"-I{isa}/include",
            str(cpu),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True)
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativeGemmN32(__gm__ int64_t *args) {
 auto*a=reinterpret_cast<__gm__ Tensor*>(args[0]);auto*b=reinterpret_cast<__gm__ Tensor*>(args[1]);auto*c=reinterpret_cast<__gm__ Tensor*>(args[2]);
 a1_n32_gemm_subview(reinterpret_cast<__gm__ half*>(a->buffer.addr)+16,reinterpret_cast<__gm__ half*>(b->buffer.addr)+16,reinterpret_cast<__gm__ float*>(c->buffer.addr)+16);
}
"""
    native = tmp_path / "native.cpp"
    native.write_text(
        generated.replace(
            'extern "C" __global__ AICORE void a1_n32_gemm_subview',
            "AICORE void a1_n32_gemm_subview",
        )
        + wrapper
    )
    image = simpler.KernelCompiler("a1").compile_single_launch(str(native), isa)
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in (1583, 1597):
            g = torch.Generator().manual_seed(seed)
            a = (torch.randint(-8, 9, (32, 64), generator=g).float() / 16).half()
            b = (torch.randint(-8, 9, (64, 32), generator=g).float() / 16).half()
            ga = torch.full((a.numel() + 32,), 7.0, dtype=torch.float16)
            ga[16:-16] = a.flatten()
            gb = torch.full((b.numel() + 32,), 7.0, dtype=torch.float16)
            gb[16:-16] = b.flatten()
            original_a, original_b = ga.clone(), gb.clone()
            result = torch.full((1024 + 32,), 7.0, dtype=torch.float32)
            runner(ga, gb, result)
            expected = (a.float() @ b.float()) * (2 if mode == "accumulate" else 1)
            assert torch.equal(result[16:-16].reshape(32, 32), expected)
            assert torch.all(result[:16] == 7) and torch.all(result[-16:] == 7)
            assert torch.equal(ga, original_a) and torch.equal(gb, original_b)
