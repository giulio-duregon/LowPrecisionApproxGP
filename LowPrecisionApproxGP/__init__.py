from . import util, model
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel, PeriodicKernel
import torch
from sklearn.model_selection import train_test_split
from LowPrecisionApproxGP.util.dtype_converter import convert_tensors_to_dtype
import pandas as pd

# Constants for dataset locations created from `scripts/getData.sh`
ENERGY_DATASET_PATH = "data/energy/energy.csv"
BIKES_DATASET_PATH = "data/bikes/hour.csv"
ROAD3D_DATASET_PATH = "data/road3d/3droad.txt"


# Helper functions for Kernel Selection (See Factories Below)
def get_base_kernel():
    return ScaleKernel(RBFKernel())


def get_periodic_kernel():
    return ScaleKernel(PeriodicKernel())


def get_matern_kernel():
    return ScaleKernel(MaternKernel())


def get_composite_kernel():
    return ScaleKernel(RBFKernel() + PeriodicKernel())


def load_bikes(dtype: torch.dtype = torch.float64, test_size: float = 0.2):
    """
    #### Loads bikes dataset from `BIKES_DATASET_PATH`, converts to a `dtype` of type `torch.dtype`
    #### Train/Test split is determined by `test_size`
    ## Return Types
    - If device == 'cpu' returns tuple of of (train),(test) data i.e. (x_train, y_train), (x_test,y_test)
    - If device == 'gpu' returns tuple of dataloaders, (train_loader, test_loader)
    """
    # Load data, get train test splits
    df = pd.read_csv(BIKES_DATASET_PATH)
    train, test = train_test_split(df, test_size=test_size)

    # Get Relevant Columns
    y_train, y_test = train["cnt"], test["cnt"]
    x_train, x_test = train.drop(["cnt", "dteday", "instant"], axis=1), test.drop(
        ["cnt", "dteday", "instant"], axis=1
    )

    # Convert to Tensors
    x_train, x_test = torch.Tensor(x_train.to_numpy()), torch.Tensor(x_test.to_numpy())
    y_train, y_test = torch.Tensor(y_train.to_numpy()), torch.Tensor(y_test.to_numpy())

    # Convert to relevant dtype
    x_train, y_train, x_test, y_test = convert_tensors_to_dtype(
        dtype, x_train, y_train, x_test, y_test
    )

    if torch.cuda.is_available():
        from torch.utils.data import TensorDataset, DataLoader

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        return train_loader, test_loader

    return (x_train, y_train), (x_test, y_test)


def load_energy(dtype: torch.dtype, test_size: float = 0.2):
    """
    #### Loads energy dataset from `ENERGY_DATASET_PATH`, converts to a `dtype` of type `torch.dtype`
    #### Train/Test split is determined by `test_size`
    ## Return Types
    - If device == 'cpu' returns tuple of of (train),(test) data i.e. (x_train, y_train), (x_test,y_test)
    - If device == 'gpu' returns tuple of dataloaders, (train_loader, test_loader)
    """
    # Load data, get train test splits
    df = pd.read_csv(ENERGY_DATASET_PATH)
    train, test = train_test_split(df, test_size=test_size)

    # Get Relevant Columns
    y_train, y_test = train[["Y1", "Y2"]], test[["Y1", "Y2"]]
    print(train.columns)
    x_train, x_test = train.drop(["Unnamed: 0", "Y1", "Y2"], axis=1), test.drop(
        ["Unnamed: 0", "Y1", "Y2"], axis=1
    )

    # Convert to Tensors
    x_train, x_test = torch.Tensor(x_train.to_numpy()), torch.Tensor(x_test.to_numpy())
    y_train, y_test = torch.Tensor(y_train.to_numpy()), torch.Tensor(y_test.to_numpy())

    # Convert to relevant dtype
    x_train, y_train, x_test, y_test = convert_tensors_to_dtype(
        dtype, x_train, y_train, x_test, y_test
    )

    if torch.cuda.is_available():
        from torch.utils.data import TensorDataset, DataLoader

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        return train_loader, test_loader

    return (x_train, y_train), (x_test, y_test)


def load_road3d(dtype: torch.dtype, test_size: float = 0.2):
    """
    #### Loads 3droad dataset from `ROAD3D_DATASET_PATH`, converts to a `dtype` of type `torch.dtype`
    #### Train/Test split is determined by `test_size`
    ## Return Types
    - If device == 'cpu' returns tuple of of (train),(test) data i.e. (x_train, y_train), (x_test,y_test)
    - If device == 'gpu' returns tuple of dataloaders, (train_loader, test_loader)
    """
    # Load data, get train test splits
    headers = {0: "OSM_ID", 1: "LONGITUDE", 2: "LATITUDE", 3: "ALTITUDE"}
    df = pd.read_csv(ROAD3D_DATASET_PATH, header=None)
    df.rename(headers, axis=1, inplace=True)
    train, test = train_test_split(df, test_size=test_size)

    # Get Relevant Columns
    y_train, y_test = train["ALTITUDE"], test["ALTITUDE"]
    x_train, x_test = train.drop(["ALTITUDE"], axis=1), test.drop(["ALTITUDE"], axis=1)

    # Convert to Tensors
    x_train, x_test = torch.Tensor(x_train.to_numpy()), torch.Tensor(x_test.to_numpy())
    y_train, y_test = torch.Tensor(y_train.to_numpy()), torch.Tensor(y_test.to_numpy())

    # Convert to relevant dtype
    x_train, y_train, x_test, y_test = convert_tensors_to_dtype(
        dtype, x_train, y_train, x_test, y_test
    )

    if torch.cuda.is_available():
        from torch.utils.data import TensorDataset, DataLoader

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

        test_dataset = TensorDataset(x_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        return train_loader, test_loader

    return (x_train, y_train), (x_test, y_test)


# Factories for model running
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
    "bikes": load_bikes,
    "energy": load_bikes,
    "road3d": load_road3d,
}

__all__ = ["util", "model", "DTYPE_FACTORY", "KERNEL_FACTORY", "DATASET_FACTORY", "load_energy", "load_bikes",
           "load_road3d"]
