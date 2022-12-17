import gpytorch
import torch
import numpy as np
import random
from LowPrecisionApproxGP import DTYPE_FACTORY, KERNEL_FACTORY, DATASET_FACTORY
from LowPrecisionApproxGP.util.GreedyTrain import greedy_train
from LowPrecisionApproxGP.model.models import VarPrecisionModel
from LowPrecisionApproxGP import (
    load_bikes,
    load_road3d,
    load_energy,
    load_naval,
    load_protein,
)
import argparse
import logging
from timeit import default_timer as timer
from datetime import timedelta
import uuid

MODEL_RND_ID = str(uuid.uuid4())


def setup_logging(logging_directory_path=None):
    """
    Helper function that sets up logging for running and recording experiment results and setting up
    a model index
    """
    # Set up default behavior if no specific logging directory was passed
    if logging_directory_path is None:
        default_logging_directory_path = "experiments"

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
    parser.add_argument("--Model_Type", default="baseline")
    parser.add_argument(
        "-d",
        "--dataset",
        default="bikes",
        type=str,
        choices=["bikes", "energy", "protein", "road3d"],
    )
    parser.add_argument(
        "-bk", "--base_kernel_type", default="base", type=str, choices=["base", ""]
    )
    parser.add_argument("-it", "--training_iter", default=100, type=int)
    parser.add_argument(
        "-dt",
        "--dtype",
        default="double",
        type=str,
        choices=["single", "double"],
    )
    parser.add_argument("-l", "--logging", type=bool)
    parser.add_argument("-s", "--seed", type=int, default=0)
    return parser.parse_args()


# This is for baseline model, Vanilla Exact GP Model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def main(logger, **kwargs):
    # Load relevant dataset w specified dtype
    training_iter = kwargs.get("training_iter")
    dtype = DTYPE_FACTORY[kwargs.get("dtype")]
    get_dataset = DATASET_FACTORY[kwargs.get("dataset")]
    train_data, test_data = get_dataset(dtype)
    train_x, train_y = train_data
    test_x, test_y = test_data

    # Create Likelihood / Model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Set to training mode
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    start_time = timer()
    with gpytorch.settings.linalg_dtypes(default=dtype):
        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            print(
                "Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f"
                % (
                    i + 1,
                    training_iter,
                    loss.item(),
                    model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item(),
                )
            )
            optimizer.step()
    end_time = timer()
    time_delta = end_time - start_time
    time_delta_formatted = timedelta(seconds=end_time - start_time)
    with torch.no_grad():
        final_train_loss = -mll(model(train_x), train_y).mean().item()
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        trained_pred_dist = likelihood(model(test_x))
        predictive_mean = trained_pred_dist.mean
        lower, upper = trained_pred_dist.confidence_region()

    final_msll = gpytorch.metrics.mean_standardized_log_loss(trained_pred_dist, test_y)
    final_mse = gpytorch.metrics.mean_squared_error(
        trained_pred_dist, test_y, squared=True
    )

    final_mae = gpytorch.metrics.mean_absolute_error(trained_pred_dist, test_y)
    logger.info(
        f"Model_ID:'{MODEL_RND_ID}', {{'Start_Time':'{start_time}', 'End_Time':'{end_time}', 'Time_Delta_Seconds':'{time_delta}','Time_Delta_Formatted':'{time_delta_formatted}','Mean_Standardized_Log_Test_Loss':'{final_msll}','Mean_Squared_Test_Error':'{final_mse}','Mean_Absolute_Test_Error':'{final_mae}','Final_Train_Loss':'{final_train_loss}'}}"
    )


if __name__ == "__main__":
    # Get argv
    args = vars(parse_args())
    seed = args.get("seed", 0)

    logger = setup_logging(args.pop("logging_output_path", None))
    logger.info(f"Model_ID:'{MODEL_RND_ID}',{args}")

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    main(logger, **args)
