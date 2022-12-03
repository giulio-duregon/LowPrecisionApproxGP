from . import util, model
from .util import *
from .model import *
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel, PeriodicKernel
import torch


def get_base_kernel():
    return ScaleKernel(RBFKernel())


def get_periodic_kernel():
    return ScaleKernel(PeriodicKernel())


def get_matern_kernel():
    return ScaleKernel(MaternKernel())


def get_composite_kernel():
    return ScaleKernel(RBFKernel() + PeriodicKernel())

def load_bikes():
    pass

def load_energy():
    pass

def load_road3d():
    pass

KERNEL_FACTORY = {
    "base": get_base_kernel,
    "composite": get_composite_kernel,
    "periodic": get_periodic_kernel,
    "matern": get_matern_kernel,
}

DTYPE_FACTORY = {
    "half": torch.float16,
    "single": torch.float32,
    "double": torch.float64,
}

DATASET_FACTORY = {
    "bikes" : load_bikes,
    "energy" : load_bikes,
    "road3d" : load_road3d,
}


__all__ = ["util", "model"]
