from datetime import date
from typing import Tuple
import torch
import logging
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
from GreedySelector import greedy_select_points
import os
from datetime import date

experiment_folder_path = os.getenv("EXPERIMENT_OUTPUTS")
if experiment_folder_path is None:
    raise ValueError(
        "No Enviroment Variable Value for EXPERIMENT_OUTPUTS, make sure to run source setup.sh"
    )

file_destination = f"{experiment_folder_path}/{date.today()}.log"
print(file_destination)
logging.basicConfig(
    filename=file_destination, filemode="a", encoding="utf-8", level=logging.INFO
)


def greedy_train(
    train_data: Tuple[torch.Tensor, torch.Tensor],
    model: ExactGP,
    mll: ExactMarginalLogLikelihood,
    max_iter: int = 50,
    max_inducing_points: int = 50,
) -> ExactGP:

    train_x, train_y = train_data

    inducing_point_candidates = train_x.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    logging.info(
        f"Starting training loop, {max_iter} max iterations, \
        {max_inducing_points} max inducing points"
    )

    for i in range(max_iter):
        # If haven't gotten any inducing points, grab a random one
        if len(inducing_point_candidates) == len(train_x):
            random_index = np.random.randint(0, len(train_x))
            first_inducing_point = (
                inducing_point_candidates[random_index].detach().clone().reshape(1, -1)
            )  # Get
            model.covar_module.inducing_points = torch.nn.Parameter(
                first_inducing_point, requires_grad=False
            )  # Set
            # Remove Selected Point from candidate set
            inducing_point_candidates = torch.cat(
                (
                    inducing_point_candidates[:random_index],
                    inducing_point_candidates[random_index + 1 :],
                ),
                dim=0,
            )

        elif len(model.covar_module.inducing_points) >= max_inducing_points:
            logging.info(
                f"Breaking out of training loop at iteration {i+1}/{max_iter}. Reached limit of inducing points: we have {len(model.covar_module.inducing_points)} \
                points with a maximum of {max_inducing_points}"
            )
            break
        else:
            inducing_point_candidates = greedy_select_points(
                model, inducing_point_candidates, train_x, train_y, mll
            )
            if inducing_point_candidates is None:
                # We've failed to find a point that increases our Likelihood
                logging.info(
                    f"Breaking out of training loop at iteration {i}/{max_iter}. Failed to add inducing point."
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

        logging.info(
            f"Iteration: {i+1}/{max_iter} - Average Loss: {loss.mean().item()}"
        )
        torch.cuda.empty_cache()

        optimizer.step()

    return model
