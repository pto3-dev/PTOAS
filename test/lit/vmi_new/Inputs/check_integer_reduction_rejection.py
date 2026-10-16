# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Reject direct eight-bit reductions at the CLI, IR and Python boundaries."""

import itertools
import subprocess
import sys

from ptoas.mlir.dialects import func
from ptoas.mlir.dialects import pto as pto_ir
from ptoas.mlir.ir import (
    Context,
    FunctionType,
    InsertionPoint,
    IntegerType,
    Location,
    Module,
    Type,
)

from ptodsl import _vmi_namespace as ns

DIAGNOSTIC = "8-bit integer reductions are not supported"
SHAPES = ((256, None), (256, 1), (256, 8), (8, 8), (1, 1), (4, 1))
OPERATIONS = (("vcadd", "addi"), ("vcmax", "maxi"), ("vcmin", "mini"))


def reduction_ir(op, element, lanes, groups, legacy=False):
    count = groups or 1
    attribute = "num_groups" if legacy else "group"
    attrs = "" if groups is None else f" {{{attribute} = {groups}}}"
    return f"""module {{
  func.func @probe(%x: !pto.vmi.vreg<{lanes}x{element}>,
                   %m: !pto.vmi.mask<{lanes}xpred>)
                   -> !pto.vmi.vreg<{count}x{element}> {{
    %r = pto.vmi.{op} %x, %m{attrs}
        : !pto.vmi.vreg<{lanes}x{element}>, !pto.vmi.mask<{lanes}xpred>
        -> !pto.vmi.vreg<{count}x{element}>
    return %r : !pto.vmi.vreg<{count}x{element}>
  }}
}}
"""


def expect_rejected(command, source):
    result = subprocess.run(
        command, input=source, text=True, capture_output=True, timeout=30, check=False
    )
    assert result.returncode > 0, (command, result.returncode, result.stderr, source)
    assert DIAGNOSTIC in result.stderr, (command, result.stderr, source)
    assert "vmi-legalize-integer-reductions" not in result.stderr


def check_compilers(opt, cli):
    cli_command = [
        cli,
        "--pto-arch=a5",
        "--pto-level=level3",
        "--pto-backend=vpto",
        "--emit-vpto",
        "-",
        "-o",
        "/dev/null",
    ]
    for element, (public, legacy), (lanes, groups) in itertools.product(
        ("i8", "si8", "ui8"), OPERATIONS, SHAPES
    ):
        source = reduction_ir(public, element, lanes, groups)
        expect_rejected([opt, "-", "-vmi-to-vpto"], source)
        expect_rejected(cli_command, source)
        operation = f"reduce_{legacy}" if groups is None else f"group_reduce_{legacy}"
        source = reduction_ir(operation, element, lanes, groups, legacy=True)
        expect_rejected([opt, "-", "-vmi-to-vpto"], source)


def python_reduction(element, lanes, groups, operation):
    source = pto_ir.VMIVRegType.get(lanes, element)
    mask = Type.parse(f"!pto.vmi.mask<{lanes}xpred>")
    module = Module.create()
    with InsertionPoint(module.body):
        fn = func.FuncOp("probe", FunctionType.get([source, mask], []))
    entry = fn.add_entry_block()
    with InsertionPoint(entry):
        result = getattr(ns.vmi, operation)(*entry.arguments, group=groups)
        func.ReturnOp([])
    assert module.operation.verify()
    return ns._raw(result).type


def check_python():
    with Context() as context, Location.unknown():
        pto_ir.register_dialect(context)
        for factory, (operation, _), (lanes, groups) in itertools.product(
            (
                IntegerType.get_signless,
                IntegerType.get_signed,
                IntegerType.get_unsigned,
            ),
            OPERATIONS,
            SHAPES,
        ):
            try:
                python_reduction(factory(8), lanes, groups, operation)
            except ValueError as error:
                assert DIAGNOSTIC in str(error), str(error)
            else:
                raise AssertionError((operation, factory(8), lanes, groups))
        for width, (operation, _) in itertools.product((16, 32), OPERATIONS):
            element = IntegerType.get_unsigned(width)
            result = python_reduction(element, 64, 8, operation)
            assert result == pto_ir.VMIVRegType.get(8, element)


if __name__ == "__main__":
    check_compilers(*sys.argv[1:])
    check_python()
    print("162 compiler rejections, 54 Python rejections, 6 supported Python cases")
