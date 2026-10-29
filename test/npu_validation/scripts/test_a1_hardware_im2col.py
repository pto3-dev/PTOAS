# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-first test of generated A1 image DMA, im2col and Cube dependencies."""

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


@pytest.mark.parametrize("profile", ["single", "reuse", "double_buffer"])
def test_a1_hardware_im2col_emitted_cpp(tmp_path, profile):
    root = Path(__file__).resolve().parents[3]
    isa = os.environ.get("PTO_ISA_ROOT")
    if not isa:
        pytest.skip("PTO_ISA_ROOT must select the A1 strided image load")
    compiler = shutil.which("clang++-19")
    assert compiler, "CPU oracle requires Clang19"
    cpp = tmp_path / "im2col.cpp"
    suffix = "" if profile == "single" else f"_{profile}"
    name = f"a1_hardware_im2col{suffix}"
    source = f"hardware_im2col{suffix}.pto"
    reuse = profile != "single"
    subprocess.run(
        [
            str(root / "build/tools/ptoas/ptoas"),
            "--pto-arch=a1",
            *(["--pto-level=level3"] if profile == "double_buffer" else []),
            str(root / "test/lit/a1" / source),
            "-o",
            str(cpp),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    generated = cpp.read_text()
    assert "TIMG2COL<" in generated and "TMATMUL(" in generated
    assert "set_flag(PIPE_MTE2, PIPE_MTE1," in generated
    assert "set_flag(PIPE_MTE1, PIPE_M," in generated
    assert "set_flag(PIPE_M, PIPE_V," in generated
    assert "set_flag(PIPE_V, PIPE_MTE3," in generated
    if reuse:
        assert generated.count("TIMG2COL<") == 2 and "TMATMUL_ACC(" in generated
        if profile == "reuse":
            assert "set_flag(PIPE_M, PIPE_MTE1," in generated
        else:
            between = generated.split("TMATMUL(", 1)[1].split("TIMG2COL<", 1)[0]
            assert "wait_flag(PIPE_M, PIPE_MTE1," not in between
    cpu_main = r"""
#include <vector>
int main() {
    std::vector<half> input(8*4*64), weight(288*16);
    std::vector<float> out(16*16+32,7);
    for (int i=0;i<8*4*64;++i) input[i]=half((i%13-6)/16.0f);
    for (int i=0;i<288*16;++i) weight[i]=half((i%11-5)/16.0f);
    a1_hardware_im2col(out.data()+16,input.data(),weight.data());
    for(int m=0;m<16;++m) for(int n=0;n<16;++n) {
        float expected=0;
        for(int c1=0;c1<2;++c1) for(int fy=0;fy<3;++fy) for(int fx=0;fx<3;++fx)
            for(int c0=0;c0<16;++c0) {
                const int y=m/4+fy-1, x=m%4+fx-1;
                if(y>=0 && y<4 && x>=0 && x<4) {
                    const int k=(c1*9+fy*3+fx)*16+c0;
                    expected+=float(input[((y+2)*4+x)*64+16+c1*16+c0])*float(weight[k*16+n]);
                }
            }
        if(out[16+m*16+n]!=expected) return 1;
    }
    for(int i=0;i<16;++i) if(out[i]!=7 || out[272+i]!=7) return 2;
    return 0;
}
"""
    cpu_main = cpu_main.replace("a1_hardware_im2col", name).replace(
        "!=expected", "!=2*expected" if reuse else "!=expected"
    )
    cpu = tmp_path / "cpu.cpp"
    cpu.write_text("#include <cstdint>\n" + generated + cpu_main)
    executable = tmp_path / "cpu-im2col"
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
extern "C" __global__ AICORE void NativeImageConvolution(__gm__ int64_t* args) {
    auto* input=reinterpret_cast<__gm__ Tensor*>(args[0]);
    auto* weight=reinterpret_cast<__gm__ Tensor*>(args[1]);
    auto* out=reinterpret_cast<__gm__ Tensor*>(args[2]);
    a1_hardware_im2col(reinterpret_cast<__gm__ float*>(out->buffer.addr)+16,
                      reinterpret_cast<__gm__ half*>(input->buffer.addr)+16,
                      reinterpret_cast<__gm__ half*>(weight->buffer.addr)+16);
}
"""
    wrapper = wrapper.replace("a1_hardware_im2col", name)
    native = tmp_path / "native.cpp"
    body = generated.replace(
        f'extern "C" __global__ AICORE void {name}', f"AICORE void {name}"
    )
    assert body != generated
    native.write_text(body + wrapper)
    image = simpler.KernelCompiler("a1").compile_single_launch(str(native), isa)
    with simpler.A1SingleLaunch(image, device_id=0) as runner:
        for seed in (971, 977):
            g = torch.Generator().manual_seed(seed)
            source = (torch.randint(-8, 9, (8, 4, 64), generator=g).float() / 16).half()
            weight = (torch.randint(-8, 9, (288, 16), generator=g).float() / 16).half()
            guarded = torch.full((source.numel() + 32,), 7.0, dtype=torch.float16)
            guarded[16:-16] = source.flatten()
            original = guarded.clone()
            weights = torch.full((weight.numel() + 32,), 7.0, dtype=torch.float16)
            weights[16:-16] = weight.flatten()
            original_weights = weights.clone()
            output = torch.full((16 * 16 + 32,), 7.0, dtype=torch.float32)
            runner(guarded, weights, output)
            x = source[2:6, :, 16:48].permute(2, 0, 1).unsqueeze(0)
            patches = torch.nn.functional.unfold(x.float(), 3, padding=1)
            patches = (
                patches.reshape(1, 2, 16, 3, 3, 16)
                .permute(0, 5, 1, 3, 4, 2)
                .reshape(16, 288)
            )
            assert torch.equal(
                output[16:-16].reshape(16, 16),
                (2 if reuse else 1) * (patches @ weight.float()),
            )
            assert torch.equal(guarded, original) and torch.equal(
                weights, original_weights
            )
            assert torch.all(output[:16] == 7) and torch.all(output[-16:] == 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
