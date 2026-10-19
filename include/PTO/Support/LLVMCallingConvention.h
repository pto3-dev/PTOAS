// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_SUPPORT_LLVM_CALLING_CONVENTION_H
#define PTO_SUPPORT_LLVM_CALLING_CONVENTION_H

#include "llvm/IR/CallingConv.h"

namespace mlir::pto {

inline bool isSimtEntryCallingConvention(llvm::CallingConv::ID convention) {
#ifdef PTOAS_EMITC_ONLY_BUILD
  // The driver rejects VPTO in this build profile. Upstream LLVM does not
  // define SimtEntry, so no SIMT caller/callee annotation can be collected.
  (void)convention;
  return false;
#else
  return convention == llvm::CallingConv::SimtEntry;
#endif
}

} // namespace mlir::pto

#endif // PTO_SUPPORT_LLVM_CALLING_CONVENTION_H
