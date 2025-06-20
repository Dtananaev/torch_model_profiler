from collections import deque
from typing import Any
import torch
import torch.nn as nn



def flatten(inputs: Any)-> list[torch.Tensor]:
    """Flatten nested structures of tensors into a flat list."""
    queue = deque([inputs])
    outputs = []
    while queue:
        x = queue.popleft()
        if isinstance(x, (list, tuple)):
            queue.extend(x)
        elif isinstance(x, dict):
            queue.extend(x.values())
        elif isinstance(x, torch.Tensor):
            outputs.append(x)
    return outputs


class Flatten(nn.Module):
    """A module that flattens nested structures of tensors into a flat list."""
    def __init__(self, model: nn.Module)-> None:
        """Initialize the Flatten module with a model."""
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs)-> list[torch.Tensor]:
        """Forward pass that flattens the outputs of the model."""
        outputs = self.model(*args, **kwargs)
        return flatten(outputs)
