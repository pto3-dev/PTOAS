# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU and 310P verification of bounded rotating L1 matrix slots."""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
simpler = pytest.importorskip("simpler_setup")


def test_a1_dynamic_l1_slots(tmp_path):
    try:
        acl = ctypes.CDLL("libascendcl.so")
    except OSError:
        pytest.skip("Ascend runtime unavailable")
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    soc = acl.aclrtGetSocName()
    if not soc or not soc.startswith(b"Ascend310P"):
        pytest.skip("Requires Ascend310P")
    root = Path(__file__).resolve().parents[3]
    isa = os.environ.get("PTO_ISA_ROOT")
    assert isa
    compiler = shutil.which("clang++-19")
    assert compiler
    cpp = tmp_path / "generated.cpp"
    subprocess.run(
        [
            str(root / "build/tools/ptoas/ptoas"),
            "--pto-arch=a1",
            "--pto-level=level3",
            str(root / "test/lit/a1/dynamic_l1_slots.pto"),
            "-o",
            str(cpp),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = cpp.read_text()
    assert "TMATMUL(" in generated and "TASSIGN(" in generated
    assert "set_flag(PIPE_MTE1, PIPE_MTE2," in generated
    assert "wait_flag(PIPE_MTE1, PIPE_MTE2," in generated
    cpu_main = r"""
#include <vector>
int main() {
 std::vector<half> a(80*16), b(16*16,half(0)); std::vector<float> out(80*16+32,7);
 for(int i=0;i<80*16;++i)a[i]=half((i%17-8)/16.0f);
 for(int i=0;i<16;++i)b[i*16+i]=half(1);
 a1_dynamic_l1_slots(a.data(),b.data(),out.data()+16);
 for(int i=0;i<80*16;++i)if(out[16+i]!=float(a[i]))return 1;
 for(int i=0;i<16;++i)if(out[i]!=7||out[16+80*16+i]!=7)return 2;
 return 0;
}
"""
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
extern "C" __global__ AICORE void DynamicL1Slots(__gm__ int64_t *args) {
 auto*a=reinterpret_cast<__gm__ Tensor*>(args[0]);auto*b=reinterpret_cast<__gm__ Tensor*>(args[1]);auto*c=reinterpret_cast<__gm__ Tensor*>(args[2]);
 a1_dynamic_l1_slots(reinterpret_cast<__gm__ half*>(a->buffer.addr)+16,reinterpret_cast<__gm__ half*>(b->buffer.addr)+16,reinterpret_cast<__gm__ float*>(c->buffer.addr)+16);
}
"""
    native = tmp_path / "native.cpp"
    native.write_text(
        generated.replace(
            'extern "C" __global__ AICORE void a1_dynamic_l1_slots',
            "AICORE void a1_dynamic_l1_slots",
        )
        + wrapper
    )
    image = simpler.KernelCompiler("a1").compile_single_launch(str(native), isa)
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in (1621, 1627):
            generator = torch.Generator().manual_seed(seed)
            a = (
                torch.randint(-8, 9, (80, 16), generator=generator).float() / 16
            ).half()
            b = (torch.randint(-8, 9, (16, 16), generator=generator).float() / 16).half()
            ga = torch.full((a.numel() + 32,), 7.0, dtype=torch.float16)
            gb = torch.full((b.numel() + 32,), 7.0, dtype=torch.float16)
            ga[16:-16] = a.flatten()
            gb[16:-16] = b.flatten()
            original_a, original_b = ga.clone(), gb.clone()
            result = torch.full((a.numel() + 32,), 7.0, dtype=torch.float32)
            runner(ga, gb, result)
            assert torch.equal(result[16:-16].reshape(80, 16), a.float() @ b.float())
            assert torch.all(result[:16] == 7) and torch.all(result[-16:] == 7)
            assert torch.equal(ga, original_a) and torch.equal(gb, original_b)
