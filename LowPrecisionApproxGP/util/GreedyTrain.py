from datetime import date
from typing import Tuple
import torch
import logging
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
from .GreedyMaxSelector import greedy_select_points_max
from .GreedySelector import greedy_select_points
import os
from datetime import date
import gpytorch

def greedy_train(
    train_data: Tuple[torch.Tensor, torch.Tensor],
    model: ExactGP,
    mll: ExactMarginalLogLikelihood,
    max_iter: int = 50,
    max_inducing_points: int = 50,
    model_name: str = None,
    dtype: torch.dtype = None,
    Use_Max: bool = True,               #If you want to find max or just the first increasing inducing point
    J: int = 0,                         #Use J=0 if you want to find maximizing MLL inducing point over all candidates
    max_Js: int = 10,                   #Number of J sets you want to explore without an increasing inducing point before stopping
) -> ExactGP:

    # Create model name for logging purposes
    # TODO: Pass through model_name

    if dtype is None:
        dtype = torch.float64

    logging.info(
        f"Model : {model_name}, Message : Pre-Training model.state_dict {model.state_dict()}"
    )
    train_x, train_y = train_data

    inducing_point_candidates = train_x.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    logging.info(
        f"Model : {model_name}, Message : Starting training loop, {max_iter} max iterations, \
        {max_inducing_points} max inducing points"
    )

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
                        inducing_point_candidates[random_index + 1 :],
                    ),
                    dim=0,
                )

            elif len(model.covar_module.inducing_points) >= max_inducing_points:
                logging.info(
                    f"Model : {model_name}, Message : Breaking out of training loop at iteration {i+1}/{max_iter}. Reached limit of inducing points: we have {len(model.covar_module.inducing_points)} \
                    points with a maximum of {max_inducing_points}"
                )
                break
            else:
                if Use_Max:
                    inducing_point_candidates = greedy_select_points_max(
                        model, inducing_point_candidates, train_x, train_y, mll, J
                    )
                else:
                    inducing_point_candidates = greedy_select_points(
                        model, inducing_point_candidates, train_x, train_y, mll
                    )

                if inducing_point_candidates is None:
                    if J != 0 and Use_Max:
                        # In the J set of points we chose we did not see any increase in the MLL
                        Js += 1
                        # if we have not seen an increasing inducing point in all the J sets, then break.
                        if Js > max_Js:
                            logging.info(
                                f"Model : {model_name}, Message : Breaking out of training loop at iteration {i}/{max_iter}. Failed to add inducing point in max number of J sets."
                            )
                            break
                    # We've failed to find a point that increases our Likelihood
                    logging.info(
                        f"Model : {model_name}, Message : Breaking out of training loop at iteration {i}/{max_iter}. Failed to add inducing point."
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
                f"Model : {model_name}, Message : Iteration: {i+1}/{max_iter} - Average Loss: {loss.mean().item()}"
            )
            torch.cuda.empty_cache()

            optimizer.step()

    return model
