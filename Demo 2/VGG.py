import torch.nn as nn
import torch

# Model
# Visual Graphics Group VGG with x layers
# "M" is max pooling
cfg = {
    "VGG9": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"],
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG11_2": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 1024, 1024, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}

# Decalre module and inherit from nn.Module


class VGG(nn.Module):
    def __init__(self, vgg_name, in_channels, num_classes=10):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = self._make_layers(cfg[vgg_name], self.in_channels)

        # Calculate the correct input size for the linear classifier
        dummy_input = torch.zeros(1, in_channels, 32, 32)  # Adjust input size as needed
        dummy_output = self.features(dummy_input)
        num_features = dummy_output.size(1)

        self.classifier = nn.Linear(num_features, self.num_classes)

    def forward(self, x):  # forward pass of the model
        out = self.features(x)
        out = out.view(out.size(0), -1)  # view as 1D
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
