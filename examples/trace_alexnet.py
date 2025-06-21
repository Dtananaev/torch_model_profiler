import torch
from torch import nn
from torchinspect.profile import profile_macs, profile_activations_data_movement

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


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),  # <- Added for flexibility
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000)
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':

    # Load pretrained AlexNet
    model = AlexNet()
    model.eval()
    inputs = torch.randn(1, 3, 224, 224)

    macs = profile_macs(model, inputs)
    print('{}: {:.4g} G'.format("alexnet", macs / 1e9))
    # The size of weights
    total_params = sum(p.numel() for p in model.parameters())
    print('{}: {:.4g} M'.format("alexnet", total_params / 1e6))
    weights_size = [sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)] # Convert to MB
    
    # for weight in model.parameters():
    #     if weight.requires_grad:
    #         print(f"Weight name: {weight.name}, shape: {weight.shape}, size: {weight.numel() * weight.element_size() / (1024 ** 2):.4f} MB")


    activation_data_mb, activation_data_param = profile_activation_memory(model, inputs)
    print("Activations memory usage:")

    for entry in activation_data_mb:
        print(f"{entry['layer']:<20} | Input: {entry['input_MB']:.4f} MB | Output: {entry['output_MB']:.4f} MB | Total: {entry['total_MB']:.4f} MB ")


    for entry in activation_data_param:
        print(f"{entry['layer']:<20} | Input: {entry['input_params']:.4f} | Output: {entry['output_params']:.4f} | Total: {entry['total_params']:.4f}")


    # activations = profile_activations_data_movement(model, inputs)
    # print("Activations data movement:")
    # for name, size in activations.items():
    #     print(f"{name}: {size:.4f} MB")