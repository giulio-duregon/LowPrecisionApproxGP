import argparse
import logging
import os
from timeit import default_timer as timer
from datetime import timedelta
from LowPrecisionApproxGP.util import save_model, GreedyTrain
import numpy as np
import uuid
import torch
from LowPrecisionApproxGP import DTYPE_FACTORY, KERNEL_FACTORY, DATASET_FACTORY
import gpytorch
from LowPrecisionApproxGP.model.models import VarPrecisionModel
import random


# Set up unique model id
MODEL_RND_ID = str(uuid.uuid4())
DEFAULT_MODEL_SAVE_PATH = f"{os.getcwd()}/{MODEL_RND_ID}"


def setup_logging(logging_directory_path=None):
    """
    Helper function that sets up logging for running and recording experiment results and setting up
    a model index
    """
    # Set up default behavior if no specific logging directory was passed
    if logging_directory_path is None:
        default_logging_directory_path = os.getenv("EXPERIMENT_OUTPUTS")

        if default_logging_directory_path is None:
            raise ValueError(
                "No Environment Variable Value for EXPERIMENT_OUTPUTS, make sure to run source setup.sh"
            )
        else:
            logging_directory_path = default_logging_directory_path

    # Format save path to todays date
    output_path = f"{logging_directory_path}/Model_Index.log"
    print(f"Saving Model ID/Params to Logging Output to -> {output_path}")
    loggerName = "model_runner.py"
    logFormatter = logging.Formatter(fmt="%(asctime)s - %(message)s")

    # create logger
    logger = logging.getLogger(loggerName)
    logger.setLevel(logging.INFO)

    # create console handler
    consoleHandler = logging.FileHandler(output_path, mode="a", encoding="utf-8")
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(logFormatter)

    # Add console handler to logger
    logger.addHandler(consoleHandler)
    return logger


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
        choices=["bikes", "naval", "energy", "protein", "road3d"],
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
    parser.add_argument("-sd", "--seed", type=int, default=0)
    return parser.parse_args()


def main(logger, **kwargs):
    x_train: torch.Tensor
    y_train: torch.Tensor
    # Get device, make sure we're not running in half precision if on cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if kwargs.get("dtype") == "half" and device.type == "cpu":
        raise ValueError("Cannot Select Half Precision for running on cpu")

    # Overwrite datatype string argument to torch.dtype object
    torch_dtype = DTYPE_FACTORY[kwargs["dtype"]]

    # Get Data
    train_data, test_data = DATASET_FACTORY[kwargs.get("dataset")](torch_dtype)
    x_train, y_train = train_data
    x_test, y_test = test_data

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    base_kernel = KERNEL_FACTORY[kwargs.get("base_kernel_type")]

    # Set up Likelihood, mean/covar module, and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = VarPrecisionModel(
        x_train,
        y_train,
        likelihood=likelihood,
        dtype=torch_dtype,
        mean_module=gpytorch.means.ConstantMean(),
        base_kernel=base_kernel(),
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
        use_max=kwargs.get("use_max"),
        j=kwargs.get("j"),
        max_js=kwargs.get("max_js"),
    )
    end_time = timer()
    time_delta = end_time - start_time
    time_delta_formatted = timedelta(seconds=end_time - start_time)
    with torch.no_grad():
        final_train_loss = -mll(model(x_train), y_train).mean().item()
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        trained_pred_dist = likelihood(model(x_test))
        predictive_mean = trained_pred_dist.mean
        lower, upper = trained_pred_dist.confidence_region()

    final_msll = gpytorch.metrics.mean_standardized_log_loss(trained_pred_dist, y_test)
    final_mse = gpytorch.metrics.mean_squared_error(
        trained_pred_dist, y_test, squared=True
    )

    final_mae = gpytorch.metrics.mean_absolute_error(trained_pred_dist, y_test)
    logger.info(
        f"Model_ID:{MODEL_RND_ID}, {{'Start_Time':'{start_time}', 'End_Time':'{end_time}', 'Time_Delta_Seconds':'{time_delta}','Time_Delta_Formatted':'{time_delta_formatted}','Mean_Standardized_Log_Test_Loss':'{final_msll}','Mean_Squared_Test_Error':'{final_mse}','Mean_Absolute_Test_Error':'{final_mae}','Final_Train_Loss':'{final_train_loss}'}}"
    )

    # Save Model if Applicable
    if kwargs.get("save_model"):
        save_model.save_model(
            model, kwargs.get("save_model_file_path", DEFAULT_MODEL_SAVE_PATH)
        )


if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())

    # Set up logging
    logger = setup_logging(args.pop("logging_output_path", None))
    logger.info(f"Model_ID:{MODEL_RND_ID},{args}")

    print(f"Running Model ID: {MODEL_RND_ID}")
    # Execute main
    seed = args.get("seed", 0)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    main(logger, **args)
