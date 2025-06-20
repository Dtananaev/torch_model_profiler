import torch
from torch import nn
from torchinspect.profile import profile_macs, profile_activations_data_movement


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
    
    for weight in model.parameters():
        if weight.requires_grad:
            print(f"Weight name: {weight.name}, shape: {weight.shape}, size: {weight.numel() * weight.element_size() / (1024 ** 2):.4f} MB")

    activations = profile_activations_data_movement(model, inputs)
    print("Activations data movement:")
    for name, size in activations.items():
        print(f"{name}: {size:.4f} MB")