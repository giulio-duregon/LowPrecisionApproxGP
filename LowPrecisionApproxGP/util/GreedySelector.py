import numpy as np
import torch
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from typing import Union


def greedy_select_points(
    model: ExactGP,
    inducing_point_candidates: torch.Tensor,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    mll: ExactMarginalLogLikelihood,
) -> Union[None, torch.Tensor]:
    """
    ###  Appends a point in-place to a model's inducing points. Uses greedy selection to select next point.

    ## Procedure:
    - Grab a random index from the candidate inducing points
    - - Add the point corresponding to the random index from the set of candidate inducing points to the model's actual inducing points.
    - - If the MLL is increased, we keep the point in the model's inducing points, and remove it from the candidate set.
    - - Otherwise we continue with our search, and return the previous candidate point back into the set of candidate inducing points
    - If we've successfully attained an inducing point, break and return the remaining candidate set of inducing points
    - If no point was selected, break out of training and return None

    Output: inducing_point_candidates - Tensor or None, Returns remaining candidate set of inducing points
    if a point was selected, otherwise returns none.
    """
    random_indices = np.random.permutation(len(inducing_point_candidates))
    inducing_points = model.covar_module.inducing_points

    # Get MLL from current inducing points
    with torch.no_grad():
        output = model(train_x)
        current_model_mll = mll(output, train_y)


    # While we haven't found a point
    for index in random_indices:
        rnd_point = inducing_point_candidates[index].reshape(1, -1)

        # Grab a point at random, calculate its likelihood
        temp = torch.cat((inducing_points, rnd_point), dim=0)

        # Update the inducing point kernel
        model.covar_module.inducing_points = torch.nn.Parameter(
            temp, requires_grad=False
        )

        # Get MLL for model with candidate inducing point
        with torch.no_grad():
            rnd_point_mll = mll(model(train_x), train_y)

        # If we've increased our likelihood, we've found our point
        # TODO: Validate that this approach is correct
        if rnd_point_mll.sum() > current_model_mll.sum():
            # Catch edge case where we grab the last index
            if index + 1 == len(inducing_point_candidates):
                return inducing_point_candidates[0:index]
            else:
                return torch.cat(
                    (
                        inducing_point_candidates[0:index],
                        inducing_point_candidates[index + 1 :],
                    ),
                    dim=0,
                )

    # If we couldn't increase our likelihood, get rid of the last appended inducing point
    model.covar_module.inducing_points = torch.nn.Parameter(
        temp[:-1], requires_grad=False
    )
    return None
