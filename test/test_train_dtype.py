import LowPrecisionApproxGP
import pytest
import torch


def test_greedy_train_dtypes(data, dtype: torch.Tensor):
    data = data.type(dtype)
