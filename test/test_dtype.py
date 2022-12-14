import pytest
import torch
import gpytorch
from LowPrecisionApproxGP.util.dtype_converter import convert_tensors_to_dtype
from .model_tester import ModelTester
from LowPrecisionApproxGP.util.GreedyTrain import greedy_train
from LowPrecisionApproxGP.model.inducing_point_kernel import (
    VarPrecisionInducingPointKernel,
)
from LowPrecisionApproxGP import load_road3d, load_bikes, load_energy

# torch.float16 not possible for cpu only
dtypes = (
    [torch.float32, torch.float64]

)
testdata = [
    (torch.Tensor([1, 2, 3]), torch.Tensor([0, 0, 0, 0, 0, 1])),
    (torch.Tensor([[1, 2, 3], [1, 2, 3]])),
]


# TODO: Parameterize mean / covar module with more specific versions
mean_modules = [(gpytorch.means.ConstantMean())]
covar_modules = [
    (gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())),
]


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("tensors", testdata)
def test_dtype(dtype, tensors):
    for tensor in convert_tensors_to_dtype(dtype, *tensors):
        assert tensor.dtype == dtype


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("mean_module", mean_modules)
@pytest.mark.parametrize("covar_module", covar_modules)
def test_train_dtypes(dtype, mean_module, covar_module):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train, y_train = torch.rand(100, 2), torch.rand(100)
    x_train, y_train = convert_tensors_to_dtype(dtype,x_train,y_train)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    covar_module = VarPrecisionInducingPointKernel(
        covar_module,
        inducing_points=torch.empty(1, device=device, dtype=dtype),
        likelihood=likelihood,
        dtype=dtype,
    )
    model = ModelTester(x_train, y_train, likelihood, dtype, mean_module, covar_module)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    if device == "cuda":
        model.cuda()
        likelihood.cuda()
    
    model.train()
    likelihood.train()
    model = greedy_train(
        (x_train, y_train),
        model,
        mll,
        5,
        10,
        dtype=dtype,
        use_max=True,
        j=20,
        max_js=10,
    )

    assert x_train.dtype == dtype
    assert y_train.dtype == dtype
    assert model.covar_module.inducing_points.dtype == dtype


@pytest.mark.parametrize("dtype", dtypes)
def test_load_bikes(dtype):
    if torch.cuda.is_available():
        train_dataloader, test_dataloader = load_bikes(dtype)
        x_train, y_train = train_dataloader.dataset[:10]
        x_test, y_test = test_dataloader.dataset[:10]

        assert x_train.dtype == dtype
        assert y_train.dtype == dtype
        assert x_test.dtype == dtype
        assert y_test.dtype == dtype

    else:
        train, test = load_bikes(dtype)
        x_train, y_train = train
        x_test, y_test = test
        assert x_train.dtype == dtype
        assert y_train.dtype == dtype
        assert x_test.dtype == dtype
        assert y_test.dtype == dtype


@pytest.mark.parametrize("dtype", dtypes)
def test_load_energy(dtype):
    if torch.cuda.is_available():
        train_dataloader, test_dataloader = load_energy(dtype)
        x_train, y_train = train_dataloader.dataset[:10]
        x_test, y_test = test_dataloader.dataset[:10]

        assert x_train.dtype == dtype
        assert y_train.dtype == dtype
        assert x_test.dtype == dtype
        assert y_test.dtype == dtype

    else:
        train, test = load_energy(dtype)
        x_train, y_train = train
        x_test, y_test = test
        assert x_train.dtype == dtype
        assert y_train.dtype == dtype
        assert x_test.dtype == dtype
        assert y_test.dtype == dtype


@pytest.mark.parametrize("dtype", dtypes)
def test_load_3droad(dtype):
    if torch.cuda.is_available():
        train_dataloader, test_dataloader = load_road3d(dtype)
        x_train, y_train = train_dataloader.dataset[:10]
        x_test, y_test = test_dataloader.dataset[:10]

        assert x_train.dtype == dtype
        assert y_train.dtype == dtype
        assert x_test.dtype == dtype
        assert y_test.dtype == dtype

    else:
        train, test = load_road3d(dtype)
        x_train, y_train = train
        x_test, y_test = test
        assert x_train.dtype == dtype
        assert y_train.dtype == dtype
        assert x_test.dtype == dtype
        assert y_test.dtype == dtype
