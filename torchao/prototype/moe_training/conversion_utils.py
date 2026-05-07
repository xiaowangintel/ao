# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Callable, Optional, Type, Union

import torch
from torch import nn

from torchao.prototype.moe_training.config import (
    Float8TrainingOpConfig,
    Float8DynamicActivationFloat8WeightMoEConfig,
    InferenceOpBaseConfig,
    MXFP8TrainingOpConfig,
    TrainingOpBaseConfig,
)

logger: logging.Logger = logging.getLogger(__name__)


def _get_tensor_cls_for_config(
    config: Union[TrainingOpBaseConfig, InferenceOpBaseConfig],
) -> Type[torch.Tensor]:
    """
    Returns the appropriate tensor class for the given config.
    """
    from torchao.prototype.moe_training.tensor import (
        Float8TrainingWeightWrapperTensor,
        Float8InferenceWeightWrapperTensor,
        MXFP8TrainingWeightWrapperTensor,
    )

    if isinstance(config, Float8DynamicActivationFloat8WeightMoEConfig):
        return Float8InferenceWeightWrapperTensor
    elif isinstance(config, MXFP8TrainingOpConfig):
        from torch.distributed.tensor import _dispatch, _ops

        pytorch_version_supported = hasattr(
            _ops, "scaled_mm_single_dim_strategy"
        ) and hasattr(_dispatch, "is_pinned_handler")

        assert pytorch_version_supported, (
            "Please install the latest torch nightly build to use MXFP8 training"
        )

        return MXFP8TrainingWeightWrapperTensor
    elif isinstance(config, Float8TrainingOpConfig):
        return Float8TrainingWeightWrapperTensor
    else:
        raise ValueError(f"Unsupported config type: {type(config)}")


def _swap_params(
    module: nn.Module,
    *,
    module_filter_fn: Optional[Callable[[nn.Module, str], bool]] = None,
    config: Optional[Union[TrainingOpBaseConfig, InferenceOpBaseConfig]] = None,
    target_parameter_name: Optional[str] = None,
) -> nn.Module:
    """
    Recurses through the nn.Module, recursively swapping the data tensor of
    each nn.Parameter with a training tensor subclass. Only applies if the module
    passed the module_filter_fn, if specified.

    The tensor class used is determined by the config type:
    - Float8TrainingWeightWrapperTensor for Float8TrainingOpConfig
    - MXFP8TrainingWeightWrapperTensor for MXFP8TrainingOpConfig
    - Float8InferenceWeightWrapperTensor for Float8DynamicActivationFloat8WeightMoEConfig

    Args:
        module: Module to modify.
        module_filter_fn: If specified, only the `torch.nn.Parameter` subclasses that
            that pass the filter function will be swapped. The inputs to the
            filter function are the module instance, and the FQN.

    Returns:
     nn.Module: The modified module with swapped linear layers.
    """
    from torchao.prototype.moe_training.tensor import (
        InferenceWeightWrapperBaseTensor,
        TrainingWeightWrapperBaseTensor,
    )

    tensor_cls = _get_tensor_cls_for_config(config)
    is_training = isinstance(config, TrainingOpBaseConfig)

    def _create_new_tensor(param_data):
        if not is_training:
            return tensor_cls.from_hp(param_data, config)
        return tensor_cls(param_data, config)

    if isinstance(module, nn.Parameter) and (
        module_filter_fn is None or module_filter_fn(module, "")
    ):
        if len(list(module.children())) > 0:
            raise AssertionError(
                f"Does not support a root nn.Parameter with children: {module}"
            )
        if not isinstance(module.data, (TrainingWeightWrapperBaseTensor, InferenceWeightWrapperBaseTensor)):
            new_data = _create_new_tensor(module.data)
            grad = module.requires_grad if is_training else False
            return nn.Parameter(new_data, requires_grad=grad)
        return module

    root_module = module

    def post_order_traversal(
        module: nn.Module,
        cur_fqn: Optional[str] = None,
        parent_module: Optional[nn.Module] = None,
    ):
        if cur_fqn is None:
            cur_fqn = ""

        for child_module_name, child_module in module.named_children():
            if cur_fqn == "":
                new_fqn = child_module_name
            else:
                new_fqn = f"{cur_fqn}.{child_module_name}"

            post_order_traversal(child_module, new_fqn, module)
        if module_filter_fn is None or module_filter_fn(module, cur_fqn):
            for param_name, param in module.named_parameters(recurse=False):
                if (
                    target_parameter_name is not None
                    and param_name != target_parameter_name
                ):
                    continue
                if not isinstance(param.data, (TrainingWeightWrapperBaseTensor, InferenceWeightWrapperBaseTensor)):
                    if not is_training and param.data.ndim != 3:
                        logger.info(
                            f"Skipping {cur_fqn}.{param_name} "
                            f"({param.data.ndim}D) — inference requires 3D"
                        )
                        continue
                    new_data = _create_new_tensor(param.data)
                    grad = param.requires_grad if is_training else False
                    new_param = nn.Parameter(new_data, requires_grad=grad)
                    setattr(module, param_name, new_param)
                    logger.info(
                        f"Swapped {cur_fqn}.{param_name} to {tensor_cls.__name__}"
                    )

    post_order_traversal(root_module)
    return root_module
