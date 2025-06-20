import torch
from torch import nn
from torchprofile import profile_macs
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import autocast



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

# if __name__ == '__main__':
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # Load pretrained AlexNet
#     model = AlexNet().to(device)
#     model.eval()
#     input_tensor = torch.randn(1, 3, 224, 224).to(device)
#     # Profiler
#     with profile(
#         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_alexnet'),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True,
#     ) as prof:
#         for step in range(11):
#             with record_function("alexnet_forward"):
#                 # Optionally use autocast for mixed precision
#                 # with autocast():
#                 _ = model(input_tensor)

#         # Print profiling summary
#         print(prof.key_averages().table(
#             sort_by="cuda_memory_usage", row_limit=10
#         ))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load pretrained AlexNet
    model = AlexNet().to(device)
    model.eval()
    inputs = torch.randn(5, 3, 224, 224).to(device)
    with profile(activities=[ProfilerActivity.CUDA],

            profile_memory=True, record_shapes=True) as prof:
        model(inputs)

    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))