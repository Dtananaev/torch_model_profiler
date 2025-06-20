import torch
from torchvision import models
from torchinspect.utils.trace import trace
from torchinspect.profile import profile_macs, profile_activations_data_movement
import numpy as np

if __name__ == '__main__':
    for name, model in models.__dict__.items():
        if not name.islower() or name.startswith('__') or not callable(model):
            continue
        model_name = name.lower()

        model = model().eval()
        if 'inception' not in name:
            inputs = torch.randn(1, 3, 224, 224)
        else:
            inputs = torch.randn(1, 3, 299, 299)


        macs = profile_macs(model, inputs)
        print('{}: {:.4g} G'.format(f"{model_name}", macs / 1e9))
        # The size of weights
        total_params = sum(p.numel() for p in model.parameters())
        print('{}: {:.4g} M'.format(f"{model_name}", total_params / 1e6))
        weights_size = [sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)] # Convert to MB
        print(f"Total weights size for {model_name}: {weights_size[0]:.4f} MB")
        

        # for weight in model.parameters():
        #     if weight.requires_grad:
        #         print(f"Weight name: {weight.name}, shape: {weight.shape}, size: {weight.numel() * weight.element_size() / (1024 ** 2):.4f} MB")

        activations = profile_activations_data_movement(model, inputs)

        #print("Activations data movement:")
        #for name, size in activations.items():
        #   print(f"{name}: {size:.4f} MB")

        idx = np.argmax(list(activations.values()))
        max_activation_name = list(activations.keys())[idx]
        max_activation_value = list(activations.values())[idx]
        print(f"max activation {max_activation_name}: {max_activation_value:.4f} MB")
        input(f"Model {model_name}; Press Enter to continue to the next model...")  # Pause for each model