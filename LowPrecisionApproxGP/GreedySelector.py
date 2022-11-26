# Pick first point at random
# Check if num_inducing_points >= max_num_inducing_points
# Call function to greedy select max point

# >>> # model is a gpytorch.models.ExactGP
# >>> # likelihood is a gpytorch.likelihoods.Likelihood
# >>> mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
# >>>
# >>> output = model(train_x)
# >>> loss = -mll(output, train_y)
# >>> loss.backward()


def greedy_select_points(model, input_data, train_y, likelihood, mll):
    # Get current MLL
    model_mll = likelihood(model)  # TODO
    best_point = None

    # While we haven't found a point
    while best_point is None:
        # Grab a point at random, calculate its likelihood
        rnd_point = likelihood(input_data[np.choose()])  # TODO

        # If we've increased our likelihood, we've found our point
        if rnd_point > model_mll:
            best_point = rnd_point
    return best_point
