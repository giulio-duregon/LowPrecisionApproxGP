from cmath import log
from datetime import date
from typing import Tuple
import torch
import logging
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
from .GreedyMaxSelector import greedy_select_points_max
from .GreedySelector import greedy_select_points
from datetime import date
import gpytorch
import os


def get_training_logger(logging_output_path=None, model_name=None) -> logging.Logger:
    """
    Helper function to setup training logger. Records training progress output to /Experiments/Model_Name.log
    """
    if logging_output_path is None:
        logging_output_path = (
                os.getenv("EXPERIMENT_OUTPUTS", default=(os.getcwd() + "/Experiments"))
                + "/"
                + (model_name if model_name else str(date.today())) + ".log"
        )

    logger_name = "GreedyTrain.py"
    log_formatter = logging.Formatter(fmt="%(asctime)s - %(message)s")

    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # create console handler
    consoleHandler = logging.FileHandler(
        logging_output_path, mode="a", encoding="utf-8"
    )
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(log_formatter)

    # Add console handler to logger
    logger.addHandler(consoleHandler)
    return logger


def greedy_train(
        train_data: Tuple[torch.Tensor, torch.Tensor],
        model: ExactGP,
        mll: ExactMarginalLogLikelihood,
        max_iter: int = 50,
        max_inducing_points: int = 50,
        model_name: str = None,
        logging_path: str = None,
        dtype: torch.dtype = torch.float64,
        use_max: bool = True,  # If you want to find max or just the first increasing inducing point
        j: int = 0,  # Use j=0 if you want to find maximizing MLL inducing point over all candidates
        max_js: int = 10,  # Number of j sets you want to explore without an increasing inducing point before stopping
) -> ExactGP:
    # Create model name for logging purposes
    print("Getting logger")
    logger = get_training_logger(logging_path, model_name=model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_x, train_y = train_data

    inducing_point_candidates = train_x.detach().clone().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    Js = 0

    with gpytorch.settings.linalg_dtypes(default=dtype):
        for i in range(max_iter):
            # If haven't gotten any inducing points, grab a random one
            if len(inducing_point_candidates) == len(train_x):
                random_index = np.random.randint(0, len(train_x))
                first_inducing_point = (
                    inducing_point_candidates[random_index]
                    .detach()
                    .clone()
                    .reshape(1, -1)
                )  # Get
                model.covar_module.inducing_points = torch.nn.Parameter(
                    first_inducing_point, requires_grad=False
                )  # Set
                # Remove Selected Point from candidate set
                inducing_point_candidates = torch.cat(
                    (
                        inducing_point_candidates[:random_index],
                        inducing_point_candidates[random_index + 1:],
                    ),
                    dim=0,
                )

            elif len(model.covar_module.inducing_points) >= max_inducing_points:
                logger.info(
                    f"Model:{model_name}, Message:Breaking out of training loop, Iteration:{i}/{max_iter}, Reason:Reached limit of inducing points: we have {len(model.covar_module.inducing_points)} \
                            points with a maximum of {max_inducing_points}"
                )
                break
            else:
                if use_max:
                    inducing_point_candidates = greedy_select_points_max(
                        model, inducing_point_candidates, train_x, train_y, mll, j
                    )
                else:
                    inducing_point_candidates = greedy_select_points(
                        model, inducing_point_candidates, train_x, train_y, mll
                    )

                if inducing_point_candidates is None:
                    if j != 0 and use_max:
                        # In the j set of points we chose we did not see any increase in the MLL
                        Js += 1
                        # if we have not seen an increasing inducing point in all the j sets, then break.
                        if Js > max_js:
                            logger.info(
                                f"Model:{model_name}, Message:Breaking out of training loop, Iteration:{i}/{max_iter}, Reason:Failed to add inducing point in max number of J sets"
                            )
                            break
                    # We've failed to find a point that increases our Likelihood
                    logger.info(
                        f"Model:{model_name}, Message:Breaking out of training loop, Iteration:{i}/{max_iter}, Reason:Failed to add an inducing point"
                    )
                    break

            # Zero gradients from previous iteration
            optimizer.zero_grad()
            mll.zero_grad()
            # Output from model
            output = model(train_x)

            # Calc average loss and backprop gradients
            loss = -mll(output, train_y)
            loss.mean().backward()

            logger.info(
                f"{{'Model':'{model_name}', 'Iteration':'{i + 1}', 'Max_iter':'{max_iter}', 'Average_Training_Loss':'{loss.mean().item()}'}}"
            )
            torch.cuda.empty_cache()

            optimizer.step()

    return model
