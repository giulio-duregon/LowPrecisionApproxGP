import torch


def convert_tensors_to_dtype(dtype: torch.dtype, /, *tensors: torch.Tensor):
    """
    Convert an arbitrary amount of tensors to a specific data type.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    return [x.to(device=device, dtype=dtype) for x in tensors]
