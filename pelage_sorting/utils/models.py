import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Optional



def create_model(model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
                num_classes: int = 2,
                dropout: float = 0.0,
                device: str = "cuda") -> nn.Module:
    """Create and initialize the wolverines classifier"""
    
    # Create backbone
    backbone = AutoModel.from_pretrained(model_name)
    
    # Freeze backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    
    # Create classifier head
    hidden_size = backbone.config.hidden_size
    classifier_layers = []
    if dropout > 0:
        classifier_layers.append(nn.Dropout(dropout))
    classifier_layers.append(nn.Linear(hidden_size, num_classes))
    classifier = nn.Sequential(*classifier_layers)
    
    # Create complete model
    class WolverinesModel(nn.Module):
        def __init__(self, backbone, classifier):
            super().__init__()
            self.backbone = backbone
            self.classifier = classifier
            
        def forward(self, x):
            with torch.no_grad():
                outputs = self.backbone(x)
                features = outputs.last_hidden_state[:, 0, :]  # CLS token
            return self.classifier(features)
            
        def get_trainable_parameters(self):
            return self.classifier.parameters()
    
    model = WolverinesModel(backbone, classifier)
    
    # Move to device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model

