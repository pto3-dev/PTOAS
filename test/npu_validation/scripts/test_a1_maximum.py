# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-first A1 maximum lowering through the public ISA, with output guards."""

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


@pytest.mark.parametrize("fp16", [False, True])
def test_a1_maximum_emitted_cpp(tmp_path, fp16):
    root = Path(__file__).resolve().parents[3]
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT must select public A1 maximum support")
    compiler = (
        shutil.which("clang++-19") or shutil.which("g++-15") or shutil.which("g++-14")
    )
    if compiler is None:
        pytest.skip("CPU oracle requires Clang19 or GCC>=14")
    ptoas = root / "build/tools/ptoas/ptoas"
    suffix = "f16" if fp16 else "f32"
    name, element = ("a1_fp16_max", "half") if fp16 else ("a1_fp32_max", "float")
    rows = cols = 16 if fp16 else 8
    count = rows * cols
    generated_path = tmp_path / "max.cpp"
    subprocess.run(
        [
            str(ptoas),
            "--pto-arch=a1",
            str(root / f"test/lit/a1/max_vector_{suffix}.pto"),
            "-o",
            str(generated_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = generated_path.read_text()
    assert "TMAX(" in generated and "set_flag(PIPE_MTE2, PIPE_V," in generated
    assert "set_flag(PIPE_V, PIPE_MTE3," in generated
    cpu_main = (
        r"""
#include <vector>
int main() {
    std::vector<ELEMENT> lhs(COUNT), rhs(COUNT), output(COUNT+32,ELEMENT(7));
    for(int i=0;i<COUNT;++i) {
        lhs[i]=ELEMENT((i%17-8)/8.0f);
        rhs[i]=ELEMENT((i%13-6)/8.0f);
    }
    FUNCTION(output.data()+16,lhs.data(),rhs.data());
    for(int i=0;i<COUNT;++i)
        if(output[16+i]!=(lhs[i]>rhs[i]?lhs[i]:rhs[i])) return 1;
    for(int i=0;i<16;++i) if(output[i]!=ELEMENT(7) || output[16+COUNT+i]!=ELEMENT(7)) return 2;
    return 0;
}
""".replace("ELEMENT", element)
        .replace("COUNT", str(count))
        .replace("FUNCTION", name)
    )
    cpu = tmp_path / "cpu.cpp"
    cpu.write_text("#include <cstdint>\n" + generated + cpu_main)
    binary = tmp_path / "cpu-max"
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
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    wrapper = r"""
#include "tensor.h"
extern "C" __global__ AICORE void NativeMaximum(__gm__ int64_t* args) {
    auto* out=reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* lhs=reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto* rhs=reinterpret_cast<__gm__ Tensor*>(args[2]);
    FUNCTION(reinterpret_cast<__gm__ ELEMENT*>(out->buffer.addr)+16,
             reinterpret_cast<__gm__ ELEMENT*>(lhs->buffer.addr),
             reinterpret_cast<__gm__ ELEMENT*>(rhs->buffer.addr));
}
""".replace("ELEMENT", element).replace("FUNCTION", name)
    native = tmp_path / "native.cpp"
    # Adapt only the generated entry ABI for the resident Tensor wrapper.
    native_generated = generated.replace(
        f'extern "C" __global__ AICORE void {name}', f"AICORE void {name}"
    )
    assert native_generated != generated
    native.write_text(native_generated + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(native), isa)
    dtype = torch.float16 if fp16 else torch.float32
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in [967, 971]:
            generator = torch.Generator().manual_seed(seed)
            lhs = torch.randn((rows, cols), generator=generator).to(dtype)
            rhs = torch.randn((rows, cols), generator=generator).to(dtype)
            output = torch.full((count + 32,), 7.0, dtype=dtype)
            runner(output, lhs, rhs)
            torch.testing.assert_close(
                output[16:-16].reshape(rows, cols),
                torch.maximum(lhs, rhs),
                rtol=0,
                atol=0,
            )
            assert torch.all(output[:16] == 7) and torch.all(output[-16:] == 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
