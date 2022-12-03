import torch
from gpytorch.models import ExactGP


def save_model(model: ExactGP, filename: str):
    "Helper function to save an `ExactGP` model to `filename`"
    torch.save(model.state_dict(), f"{filename}.pth")
