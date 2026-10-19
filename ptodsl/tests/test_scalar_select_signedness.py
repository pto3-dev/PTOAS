#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Regression tests for the authored integer type of ``pto.select``.

``select`` coerces both arms to a signless common carrier before emitting the
op.  The authored signedness has to be restored on the result, otherwise the
value escapes as ``i32`` while everything around it is still ``si32``/``ui32``
and the first branch merge that touches it fails.
"""

import logging
import re

from ptodsl import pto

# The restore is emitted right after the select: its result is wrapped in a
# conversion cast back to the authored signedness.
_SIGNED_RESTORE = re.compile(
    r"pto\.select[^\n]*\n"
    r"\s*%[A-Za-z0-9_]+ = builtin\.unrealized_conversion_cast %[A-Za-z0-9_]+ : i32 to si32"
)
_UNSIGNED_RESTORE = re.compile(
    r"pto\.select[^\n]*\n"
    r"\s*%[A-Za-z0-9_]+ = builtin\.unrealized_conversion_cast %[A-Za-z0-9_]+ : i32 to ui32"
)


@pto.jit(name="select_signed_branch_merge", target="a5")
def select_signed_branch_merge(out: pto.ptr(pto.si32, "gm"), value: pto.si32):
    # Shape lifted from the SIMT all-reduce leader path: the ``select`` result
    # meets a plain authored constant in a branch merge.  A signless result
    # makes ``br.assign`` report "then branch yields i32, else branch yields
    # si32".
    identity = pto.const(0, dtype=pto.si32)
    cond = value > identity
    picked = pto.select(cond, value, identity)
    with pto.if_(cond) as br:
        with br.then_:
            br.assign(result=picked)
        with br.else_:
            br.assign(result=identity)
    pto.store(br.result, out, 0)


@pto.jit(name="select_unsigned_branch_merge", target="a5")
def select_unsigned_branch_merge(out: pto.ptr(pto.ui32, "gm"), value: pto.ui32):
    identity = pto.const(0, dtype=pto.ui32)
    cond = value > identity
    picked = pto.select(cond, value, identity)
    with pto.if_(cond) as br:
        with br.then_:
            br.assign(result=picked)
        with br.else_:
            br.assign(result=identity)
    pto.store(br.result, out, 0)


@pto.jit(name="select_mixed_signedness_probe", target="a5")
def select_mixed_signedness_probe(
    out: pto.ptr(pto.si32, "gm"),
    signed_value: pto.si32,
    unsigned_value: pto.ui32,
):
    # Integer arms that disagree on type (here: signedness) are rejected —
    # silently picking one arm's signedness for the result loses the other's
    # semantics, so the author has to cast explicitly.
    cond = signed_value > pto.const(0, dtype=pto.si32)
    picked = pto.select(cond, signed_value, unsigned_value)
    with pto.if_(cond) as br:
        with br.then_:
            br.assign(result=picked)
        with br.else_:
            br.assign(result=signed_value)
    pto.store(br.result, out, 0)


@pto.jit(name="select_float_branch_merge", target="a5")
def select_float_branch_merge(out: pto.ptr(pto.f32, "gm"), value: pto.f32):
    # Floating-point arms carry no signedness; restoring must stay a no-op.
    identity = pto.const(0.0, dtype=pto.f32)
    cond = value > identity
    picked = pto.select(cond, value, identity)
    with pto.if_(cond) as br:
        with br.then_:
            br.assign(result=picked)
        with br.else_:
            br.assign(result=identity)
    pto.store(br.result, out, 0)


@pto.jit(name="select_index_integer_pair", target="a5")
def select_index_integer_pair(out: pto.ptr(pto.si32, "gm"), value: pto.si32):
    # An index/integer pair resolves to the index type; the result must stay
    # usable as a store index.
    cond = value > pto.const(0, dtype=pto.si32)
    picked = pto.select(cond, pto.const(1, dtype=pto.index), value)
    pto.store(value, out, picked)


logging.basicConfig(level=logging.INFO)


def main():
    # Compiling the branch-merge kernels is the regression: before the fix the
    # merge raised "br.assign(...) type mismatch for 'result': then branch
    # yields i32, else branch yields si32".
    signed_mlir = select_signed_branch_merge.compile().mlir_text()
    assert "pto.select" in signed_mlir
    assert _SIGNED_RESTORE.search(signed_mlir), (
        "select on two si32 arms must restore the result to si32:\n"
        + signed_mlir
    )

    unsigned_mlir = select_unsigned_branch_merge.compile().mlir_text()
    assert "pto.select" in unsigned_mlir
    assert _UNSIGNED_RESTORE.search(unsigned_mlir), (
        "select on two ui32 arms must restore the result to ui32:\n"
        + unsigned_mlir
    )

    try:
        select_mixed_signedness_probe.compile()
    except TypeError as exc:
        assert "integer arms require matching types" in str(exc)
        assert "si32" in str(exc) and "ui32" in str(exc)
    else:
        raise AssertionError(
            "select with mixed-sign integer arms must be rejected with a "
            "TypeError asking for an explicit cast"
        )

    select_float_branch_merge.compile()
    select_index_integer_pair.compile()

    logging.info("ptodsl_scalar_select_signedness: PASS")


if __name__ == "__main__":
    main()
