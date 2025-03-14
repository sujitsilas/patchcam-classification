import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from efficientnet_pytorch import EfficientNet

def get_model(model_type='inception', device='cuda'):
    """
    Returns a PyTorch model instance given a model type.
    """
    if model_type == 'inception':
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

    elif model_type == 'efficientnet':
        # Example for EfficientNet-B0
        model = EfficientNet.from_pretrained('efficientnet-b0')
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, 2)

    else:
        raise ValueError("Unknown model type specified.")

    return model.to(device)
