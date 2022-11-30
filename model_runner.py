import argparse
import logging
import os
from random import choices
from sys import argv
from LowPrecisionApproxGP.util import save_model, GreedyTrain
from datetime import date
import numpy as np
import uuid

# Set up unique model id
MODEL_RND_ID = uuid.uuid4()
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

    # Format save path to todays date
    logging_directory_path = f"{logging_directory_path}/{MODEL_RND_ID}.log"
    print(f"Saving Logging Output to -> {logging_directory_path}")

    return logging.basicConfig(
        filename=logging_directory_path,
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

    return parser.parse_args()


def main(**kwargs):
    # TODO: Dataset factory, Kernel Factory, get dataset, kernel, build model, train, save
    # TODO: Add dictionary popping + exception inside kernel creation to model args
    dataset = ...
    base_kernel_type = ...
    model = ...  # Feed in base_kernel / datatype

    # GreedyTrain.greedy_train(model, ...)

    if kwargs.get("save_model"):
        save_model(model, kwargs.get("save_model_file_path", DEFAULT_MODEL_SAVE_PATH))


if __name__ == "__main__":
    # Parse args
    args = vars(parse_args())

    # Set up logging if necessary
    if args.pop("logging", None):
        setup_logging(args.pop("logging_output_path", None))
        logging.info(f"Initializing model_runner.py with {args}")

    # Execute main
    main(**args)
