from transformers import Dinov2WithRegistersModel
from torch import nn
import torch
from math import *
from . import register_encoder


@register_encoder()
class Dinov2withNorm(nn.Module):
    def __init__(
        self,
        dinov2_path: str,
        normalize: bool = True,
        num_keep_layers: int = None,
    ):
        super().__init__()
        # Support both local paths and HuggingFace model IDs
        try:
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=True)
        except (OSError, ValueError, AttributeError):
            self.encoder = Dinov2WithRegistersModel.from_pretrained(dinov2_path, local_files_only=False)
        self.encoder.requires_grad_(False)
        if normalize:
            self.encoder.layernorm.elementwise_affine = False
            self.encoder.layernorm.weight = None
            self.encoder.layernorm.bias = None
        self.patch_size = self.encoder.config.patch_size
        self.hidden_size = self.encoder.config.hidden_size
        self.num_keep_layers = num_keep_layers
        
    def dinov2_forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(x, output_hidden_states=True)
        if self.num_keep_layers is not None:
            k_features = outputs.hidden_states[self.num_keep_layers]
            image_features = self.encoder.layernorm(k_features)
        else:
            image_features = outputs.last_hidden_state
        unused_token_num = 5
        image_features = image_features[:, unused_token_num:]
        return image_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dinov2_forward(x)
