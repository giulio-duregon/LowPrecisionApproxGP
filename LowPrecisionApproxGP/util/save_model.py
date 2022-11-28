import torch
from gpytorch.models import ExactGP


def save_model(model: ExactGP, filename: str):
    torch.save(model.state_dict(), f"{filename}.pth")
