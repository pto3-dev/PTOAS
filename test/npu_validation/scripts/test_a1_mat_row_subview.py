# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Verify dynamic C16 Mat row windows retain the parent layout and synchronization."""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
simpler = pytest.importorskip("simpler_setup")


def test_a1_mat_row_subview(tmp_path):
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
    assert isa, "PTO_ISA_ROOT must select the A1 ISA branch"
    compiler = shutil.which("clang++-19")
    assert compiler, "CPU oracle requires Clang19"
    cpp = tmp_path / "generated.cpp"
    subprocess.run(
        [
            str(root / "build/tools/ptoas/ptoas"),
            "--pto-arch=a1",
            str(root / "test/lit/a1/mat_row_subview.pto"),
            "-o",
            str(cpp),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = cpp.read_text()
    assert "TINSERT(" in generated and "TMATMUL(" in generated
    assert "set_flag(PIPE_MTE2, PIPE_MTE1," in generated
    cpu_main = r"""
#include <vector>
int main() {
 std::vector<half> a(128*16), b(16*16); std::vector<float> out(128*16+32,7);
 for(int i=0;i<128*16;++i)a[i]=half((i%19-9)/16.0f);
 for(int i=0;i<16*16;++i)b[i]=half((i%13-6)/16.0f);
 a1_mat_row_subview(out.data()+16,a.data(),b.data());
 for(int i=0;i<128;++i)for(int j=0;j<16;++j){float v=0;for(int k=0;k<16;++k)v+=float(a[i*16+k])*float(b[k*16+j]);if(out[16+i*16+j]!=v)return 1;}
 for(int i=0;i<16;++i)if(out[i]!=7||out[16+2048+i]!=7)return 2;
 return 0;
}
"""
    cpu = tmp_path / "cpu.cpp"
    cpu.write_text("#include <cstdint>\n" + generated + cpu_main)
    exe = tmp_path / "cpu"
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
            str(exe),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(exe)], check=True)
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativeMatRow(__gm__ int64_t *args) {
 auto*a=reinterpret_cast<__gm__ Tensor*>(args[0]);auto*b=reinterpret_cast<__gm__ Tensor*>(args[1]);auto*c=reinterpret_cast<__gm__ Tensor*>(args[2]);
 a1_mat_row_subview(reinterpret_cast<__gm__ float*>(c->buffer.addr)+16,reinterpret_cast<__gm__ half*>(a->buffer.addr)+16,reinterpret_cast<__gm__ half*>(b->buffer.addr)+16);
}
"""
    native = tmp_path / "native.cpp"
    native.write_text(
        generated.replace(
            'extern "C" __global__ AICORE void a1_mat_row_subview',
            "AICORE void a1_mat_row_subview",
        )
        + wrapper
    )
    image = simpler.KernelCompiler("a1").compile_single_launch(str(native), isa)
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in (4181, 4193):
            g = torch.Generator().manual_seed(seed)
            a = (torch.randint(-8, 9, (128, 16), generator=g).float() / 16).half()
            b = (torch.randint(-8, 9, (16, 16), generator=g).float() / 16).half()
            ga = torch.full((a.numel() + 32,), 7.0, dtype=torch.float16)
            gb = torch.full((b.numel() + 32,), 7.0, dtype=torch.float16)
            ga[16:-16], gb[16:-16] = a.flatten(), b.flatten()
            original_a, original_b = ga.clone(), gb.clone()
            result = torch.full((128 * 16 + 32,), 7.0, dtype=torch.float32)
            runner(ga, gb, result)
            assert torch.equal(result[16:-16].reshape(128, 16), a.float() @ b.float())
            assert torch.all(result[:16] == 7) and torch.all(result[-16:] == 7)
            assert torch.equal(ga, original_a) and torch.equal(gb, original_b)
