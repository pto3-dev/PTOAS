# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-first strict scalar arithmetic retains float cast and rounding boundaries."""

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


def test_a1_strict_scalar_emitted_cpp(tmp_path):
    root = Path(__file__).resolve().parents[3]
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT must select public A1 Tile access support")
    compiler = shutil.which("clang++-19")
    assert compiler, "CPU oracle requires Clang19"
    element, name = "float", "a1_strict_scalar"
    generated_path = tmp_path / "scalar_access.cpp"
    subprocess.run(
        [
            str(root / "build/tools/ptoas/ptoas"),
            "--pto-arch=a1",
            str(root / "test/lit/a1/strict_scalar_rounding.pto"),
            "-o",
            str(generated_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = generated_path.read_text()
    assert ".GetValue(" in generated and ".SetValue(" in generated
    assert "set_flag(PIPE_V, PIPE_S," in generated
    assert "set_flag(PIPE_S, PIPE_V," in generated
    assert "set_flag(PIPE_MTE2, PIPE_V," in generated
    assert "set_flag(PIPE_V, PIPE_MTE3," in generated
    assert generated.count("TLOAD(") == generated.count("TSTORE(") == 1
    cpu_main = r"""
#include <vector>
int main() {
    std::vector<ELEMENT> input(256), output(288, ELEMENT(7));
    for (int i=0; i<256; ++i) input[i]=ELEMENT((i%127-63)/8.0f);
    FUNCTION(output.data()+16, input.data());
    for (int i=0; i<256; ++i) if(output[16+i]!=ELEMENT((i%16==0 ? -float(int(float(input[i+1])*2*1.4426950408889634f-0.5f))*2 : float(input[i])*4))) return 1;
    for (int i=0; i<16; ++i) if(output[i]!=ELEMENT(7) || output[272+i]!=ELEMENT(7)) return 2;
    return 0;
}
""".replace("ELEMENT", element).replace("FUNCTION", name)
    cpu = tmp_path / "cpu.cpp"
    cpu.write_text("#include <cstdint>\n" + generated + cpu_main)
    binary = tmp_path / "cpu-scalar-access"
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
extern "C" __global__ AICORE void NativeScalarAccess(__gm__ int64_t* args) {
    auto* out=reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* input=reinterpret_cast<__gm__ Tensor*>(args[1]);
    FUNCTION(reinterpret_cast<__gm__ ELEMENT*>(out->buffer.addr)+16,
             reinterpret_cast<__gm__ ELEMENT*>(input->buffer.addr));
}
""".replace("ELEMENT", element).replace("FUNCTION", name)
    native = tmp_path / "native.cpp"
    native_generated = generated.replace(
        f'extern "C" __global__ AICORE void {name}', f"AICORE void {name}"
    )
    assert native_generated != generated
    native.write_text(native_generated + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(native), isa)
    dtype = torch.float32
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in [1021, 1031]:
            generator = torch.Generator().manual_seed(seed)
            values = (
                torch.randint(-256, 256, (16, 16), generator=generator).to(dtype) / 8
            )
            saved = values.clone()
            output = torch.full((288,), 7.0, dtype=dtype)
            runner(output, values)
            expected = saved * 4
            expected[:, 0] = (
                -torch.trunc(saved[:, 1] * 2 * 1.4426950408889634 - 0.5) * 2
            )
            assert torch.equal(output[16:-16], expected.reshape(-1))
            assert torch.equal(values, saved)
            assert torch.all(output[:16] == 7) and torch.all(output[-16:] == 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
