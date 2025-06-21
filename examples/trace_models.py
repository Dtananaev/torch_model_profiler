import torch
from torch import nn
from torchvision import models
from torchinspect.utils.trace import trace
from torchinspect.profile import profile_macs, profile_activations_data_movement
import numpy as np


def profile_activation_memory(model, input_tensor):
    activation_data_mb = []
    activation_data_param = []


    def hook(module, inputs, outputs):
        input_size_mb = sum(inp.numel() * inp.element_size() for inp in inputs if isinstance(inp, torch.Tensor))
        input_size_param = sum(inp.numel() for inp in inputs if isinstance(inp, torch.Tensor))

        if isinstance(outputs, (tuple, list)):
            output_size_mb = sum(out.numel() * out.element_size() for out in outputs if isinstance(out, torch.Tensor))
            output_size_param = sum(out.numel() for out in outputs if isinstance(out, torch.Tensor))

        elif isinstance(outputs, torch.Tensor):
            output_size_mb = outputs.numel() * outputs.element_size()
            output_size_param = outputs.numel() 
        else:
            output_size_mb = output_size_param = 0

        activation_data_mb.append({
            'layer': module.__class__.__name__,
            'input_MB': input_size_mb / (1024 ** 2),
            'output_MB': output_size_mb / (1024 ** 2),
            'total_MB': (input_size_mb + output_size_mb) / (1024 ** 2),
        })
        activation_data_param.append({
            'layer': module.__class__.__name__,
            'input_params': input_size_param,
            'output_params': output_size_param,
            'total_params': input_size_param + output_size_param,
        })


    hooks = []
    for module in model.modules():
        if module != model and not isinstance(module, nn.Sequential):
            hooks.append(module.register_forward_hook(hook))

    with torch.no_grad():
        _ = model(input_tensor)

    for h in hooks:
        h.remove()

    return activation_data_mb, activation_data_param

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
        activation_data_mb, activation_data_param = profile_activation_memory(model, inputs)
        print("Activations memory usage:")

        for entry in activation_data_mb:
            print(f"{entry['layer']:<20} | Input: {entry['input_MB']:.4f} MB | Output: {entry['output_MB']:.4f} MB | Total: {entry['total_MB']:.4f} MB ")


        for entry in activation_data_param:
            print(f"{entry['layer']:<20} | Input: {entry['input_params']:.4f} | Output: {entry['output_params']:.4f} | Total: {entry['total_params']:.4f}")



        input(f"Model {model_name}; Press Enter to continue to the next model...")  # Pause for each model