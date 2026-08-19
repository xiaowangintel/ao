"""Intel XPU MXTensor subclass for MX format inference."""

import torch
import torch.nn.functional as F
from torch.nn.functional import ScalingType

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import (
    MXTensor,
    QuantizeTensorToMXKwargs,
    _addmm_mx_dispatch,
    _get_gemm_choice,
    register_mx_tensor_class,
    to_mx,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

aten = torch.ops.aten


class MXTensorXPU(MXTensor):
    """MXTensor subclass for Intel XPU: no scale swizzling, device-specific gemm."""

    @staticmethod
    @torch._dynamo.allow_in_graph
    def to_mx(
        data_hp,
        elem_dtype,
        block_size=32,
        scaling_mode=ScaleCalculationMode.FLOOR,
        kernel_preference=KernelPreference.EMULATED,
        act_quant_kwargs=None,
        is_swizzled_scales=False,
        mxfp8_dim0_cast_kernel_choice=None,
    ):
        """XPU override: always is_swizzled_scales=False."""
        # Force no swizzle for XPU
        if act_quant_kwargs is not None and act_quant_kwargs.is_swizzled_scales:
            act_quant_kwargs = QuantizeTensorToMXKwargs(
                elem_dtype=act_quant_kwargs.elem_dtype,
                block_size=act_quant_kwargs.block_size,
                scaling_mode=act_quant_kwargs.scaling_mode,
                kernel_preference=act_quant_kwargs.kernel_preference,
                is_swizzled_scales=False,
            )
        scale, data_lp = to_mx(data_hp, elem_dtype, block_size, scaling_mode, False)
        return MXTensorXPU(
            data_lp,
            scale,
            elem_dtype,
            block_size,
            data_hp.dtype,
            kernel_preference,
            act_quant_kwargs,
            False,
        )


def _xpu_addmm_dispatch(a, b, aten_op, bias=None):
    """XPU-specific MX gemm dispatch."""
    if not isinstance(a, MXTensor):
        assert b.act_quant_kwargs is not None, "weight-only quant not yet supported"
        k = b.act_quant_kwargs
        a = MXTensorXPU.to_mx(
            a, k.elem_dtype, k.block_size, k.scaling_mode, k.kernel_preference
        )

    gemm_choice = _get_gemm_choice(a.kernel_preference, b.kernel_preference)

    if gemm_choice == KernelPreference.EMULATED:
        return _addmm_mx_dispatch(a, b, aten_op, bias)

    # AUTO: XPU-specific gemm
    M, K, N = a.shape[0], a.shape[1], b.shape[1]
    assert a.block_size == 32 and b.block_size == 32

    a_scale = a.scale.view(M, K // 32)
    b_scale = b.scale.t().view(N, K // 32)

    if a.elem_dtype == torch.float8_e4m3fn:
        print("====================xpu e4m3")
        assert b.elem_dtype == torch.float8_e4m3fn
        a_scale_e8m0 = a_scale.view(torch.float8_e8m0fnu)
        b_scale_e8m0 = b_scale.view(torch.float8_e8m0fnu).t().contiguous()
        return torch._scaled_mm(
            a.qdata,
            b.qdata,
            a_scale_e8m0,
            b_scale_e8m0,
            bias=bias,
            out_dtype=torch.bfloat16,
        )
    else:
        print("====================xpu e2m1")
        assert a.elem_dtype == torch.float4_e2m1fn_x2
        assert b.elem_dtype == torch.float4_e2m1fn_x2
        return F.scaled_mm(
            a.qdata.view(torch.float4_e2m1fn_x2),
            b.qdata.view(torch.float4_e2m1fn_x2),
            scale_a=a_scale,
            scale_recipe_a=ScalingType.BlockWise1x32,
            scale_b=b_scale.contiguous(),
            scale_recipe_b=ScalingType.BlockWise1x32,
            swizzle_a=None,
            swizzle_b=None,
            bias=bias,
            output_dtype=torch.bfloat16,
        )


# Override compute ops for MXTensorXPU
xpu_implements = MXTensorXPU.implements


@xpu_implements([aten.mm.default, aten.matmul.default])
def xpu_mx_mm(func, types, args, kwargs):
    return _xpu_addmm_dispatch(args[0], args[1], func)


@xpu_implements([aten.addmm.default])
def xpu_mx_addmm(func, types, args, kwargs):
    return _xpu_addmm_dispatch(args[1], args[2], func, bias=args[0])


@xpu_implements([aten.linear.default])
def xpu_mx_linear(func, types, args, kwargs):
    a = args[0]
    orig_shape = a.shape
    a_2d = a.view(-1, orig_shape[-1])
    b = args[1].t()
    bias = args[2] if len(args) > 2 else None
    if bias is not None:
        res = _xpu_addmm_dispatch(a_2d, b, aten.addmm.default, bias)
    else:
        res = _xpu_addmm_dispatch(a_2d, b, aten.mm.default)
    return res.view(*orig_shape[:-1], res.shape[-1])


# Register XPU class
register_mx_tensor_class("xpu", MXTensorXPU)

# Allow safe serialization
torch.serialization.add_safe_globals([MXTensorXPU])
