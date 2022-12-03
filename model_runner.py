import argparse
import logging
import os
from timeit import default_timer as timer
from datetime import timedelta
from LowPrecisionApproxGP.util import save_model, GreedyTrain
from LowPrecisionApproxGP.model.inducing_point_kernel import (
    VarPrecisionInducingPointKernel,
)
from datetime import date
import numpy as np
import uuid
import torch
from LowPrecisionApproxGP import DTYPE_FACTORY, KERNEL_FACTORY, DATASET_FACTORY
import gpytorch
from LowPrecisionApproxGP.model.models import VarPrecisionModel


# Set up unique model id
MODEL_RND_ID = str(uuid.uuid4())
DEFAULT_MODEL_SAVE_PATH = f"{os.getcwd()}/{MODEL_RND_ID}"


def setup_logging(logging_directory_path=None):
    """
    Helper function that sets up logging for running and recording experiment results
    """
    # Set up default behavior if no specific logging directory was passed
    if logging_directory_path is None:
        default_logging_directory_path = os.getenv("EXPERIMENT_OUTPUTS")

        if default_logging_directory_path is None:
            raise ValueError(
                "No Enviroment Variable Value for EXPERIMENT_OUTPUTS, make sure to run source setup.sh"
            )
        else:
            logging_directory_path = default_logging_directory_path

    # Format save path to todays date
    output_path = f"{logging_directory_path}/ModelIndex.log"
    print(f"Saving Model ID/Params to Logging Output to -> {output_path}")

    return logging.basicConfig(
        filename=output_path,
        filemode="a",
        encoding="utf-8",
        level=logging.INFO,
    )


def parse_args():
    """
    Helper function to set up command line argument parsing
    """
    parser = argparse.ArgumentParser()

    # Set up arguments
    parser.add_argument(
        "-d",
        "--dataset",
        default="bikes",
        type=str,
        choices=["bikes", "elevators", "energy", "road3d"],
    )
    parser.add_argument(
        "-bk", "--base_kernel_type", default="base", type=str, choices=["base", ""]
    )
    parser.add_argument("-it", "--training_iter", default=50, type=int)
    parser.add_argument("-ip", "--max_inducing_points", default=50, type=int)
    parser.add_argument(
        "-dt",
        "--dtype",
        default="double",
        type=str,
        choices=["single", "half", "double"],
    )
    parser.add_argument("-s", "--save_model", type=bool)
    parser.add_argument("-sfp", "--save_model_file_path", type=str)
    parser.add_argument("-l", "--logging", type=bool)
    parser.add_argument("-lop", "--logging_output_path", type=str)
    parser.add_argument("-m", "--use_max", type=bool, default=False)
    parser.add_argument("-j", "--j", type=int, default=0)
    parser.add_argument("-mj", "--max_js", type=int, default=10)
    return parser.parse_args()


def main(**kwargs):
    # Get device, make sure we're not running in half precision if on cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if kwargs.get("dtype") == "half" and device.type == "cpu":
        raise ValueError("Cannot Select Half Precision for running on cpu")

    # Overwrite datatype string argument to torch.dtype object
    torch_dtype = DTYPE_FACTORY[kwargs["dtype"]]

    # Get Data
    train_data, test_data = DATASET_FACTORY[kwargs.get("dataset")]()
    x_train, y_train = train_data
    base_kernel = KERNEL_FACTORY[kwargs.get("base_kernel_type")]()

    # Set up Likelihood, mean/covar module, and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    covar_module = VarPrecisionInducingPointKernel(
        base_kernel=base_kernel,
        likelihood=likelihood,
        inducing_points=torch.empty(1),
        dtype=torch_dtype,
    )
    mean_module = gpytorch.means.ConstantMean()
    model = VarPrecisionModel(
        x_train,
        y_train,
        likelihood=likelihood,
        dtype=torch_dtype,
        mean_module=mean_module,
        covar_module=covar_module,
    )

    # Set up MLL and train
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    model.train()
    likelihood.train()

    # Time Training
    start_time = timer()
    GreedyTrain.greedy_train(
        train_data=train_data,
        model=model,
        mll=mll,
        max_iter=kwargs.get("training_iter"),
        max_inducing_points=kwargs.get("max_inducing_points"),
        model_name=MODEL_RND_ID,
        dtype=torch_dtype,
        Use_Max=kwargs.get("use_max"),
        J=kwargs.get("j"),
        max_Js=kwargs.get("max_js"),
    )
    end_time = timer()
    time_delta = timedelta(seconds=end_time - start_time)
    logging.info(
        f"Model_ID:{MODEL_RND_ID}, Start_Time:{start_time}, End_Time:{end_time}, Time_Delta:{time_delta}"
    )

    # Save Model if Applicable
    if kwargs.get("save_model"):
        save_model.save_model(
            model, kwargs.get("save_model_file_path", DEFAULT_MODEL_SAVE_PATH)
        )


if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())

    # Set up logging if necessary
    if args.pop("logging", None):
        setup_logging(args.pop("logging_output_path", None))
        logging.info(f" Model_ID:{MODEL_RND_ID}, Date:{date.today()}, Args:{args}")

    # Execute main
    main(**args)
