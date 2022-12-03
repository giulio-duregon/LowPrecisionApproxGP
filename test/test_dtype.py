import pytest
import torch
from LowPrecisionApproxGP.util.dtype_converter import convert_tensors_to_dtype


dtypes = [(torch.float16), (torch.float32), (torch.float64)]
testdata = [
    (torch.Tensor([1, 2, 3]), torch.Tensor([0, 0, 0, 0, 0, 1])),
    (torch.Tensor([[1, 2, 3], [1, 2, 3]])),
]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("tensors", testdata)
def test_dtype(dtype, tensors):
    for tensor in convert_tensors_to_dtype(dtype, *tensors):
        assert tensor.dtype == dtype
