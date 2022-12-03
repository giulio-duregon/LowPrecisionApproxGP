import torch


def convert_tensors_to_dtype(dtype: torch.dtype, /, *tensors: torch.Tensor):
    """
    Convert an arbitrary amount of tensors to a specific data type.
    """
    return [x.to(dtype) for x in tensors]
