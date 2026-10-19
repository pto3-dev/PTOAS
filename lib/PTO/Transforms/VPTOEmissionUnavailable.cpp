// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Transforms/VPTOLLVMEmitter.h"

namespace mlir::pto {

static LogicalResult rejectVPTOEmission(EmittedLLVMModule &cubeModule,
                                       EmittedLLVMModule &vectorModule,
                                       llvm::raw_ostream &diagOS) {
  cubeModule.reset();
  vectorModule.reset();
  diagOS << "VPTO LLVM emission is unavailable in a PTOAS_EMITC_ONLY_BUILD "
            "compiler; rebuild with VPTO-enabled LLVM 19.\n";
  return failure();
}

LogicalResult lowerVPTOModuleToLLVMModulesBeta1(
    ModuleOp, const VPTOEmissionOptions &, EmittedLLVMModule &cubeModule,
    EmittedLLVMModule &vectorModule, llvm::raw_ostream &diagOS) {
  return rejectVPTOEmission(cubeModule, vectorModule, diagOS);
}

LogicalResult lowerVPTOModuleToLLVMModulesCANN900(
    ModuleOp, const VPTOEmissionOptions &, EmittedLLVMModule &cubeModule,
    EmittedLLVMModule &vectorModule, llvm::raw_ostream &diagOS) {
  return rejectVPTOEmission(cubeModule, vectorModule, diagOS);
}

} // namespace mlir::pto
