# src/model.py

import torch.nn as nn
import torchvision.models as models

def get_model(pretrained=True):
    """
    Carga un modelo ResNet18 y lo adapta para clasificación binaria.
    
    Args:
        pretrained (bool): Si es True, carga los pesos pre-entrenados de ImageNet.
    
    Returns:
        torch.nn.Module: El modelo de PyTorch.
    """
    weights = 'IMAGENET1K_V1' if pretrained else None
    # Cargar un ResNet18 pre-entrenado
    model = models.resnet18(weights=weights)

    # Primero congelamos todo
    for param in model.parameters():
        param.requires_grad = False

    # Descongelamos los parámetros del último bloque convolucional (layer4) y la capa final
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Reemplazar la última capa (el clasificador), que se entrenará por defecto
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    
    return model