import torch
import torch.nn as nn

from configs import Config

class ConfigurableCNN(nn.Module):
    def __init__(self, input_channels, num_classes, base_filters=64, use_bn=False):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.use_bn = use_bn

    def conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not self.use_bn)]
        if self.use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

class ClassicCNN(ConfigurableCNN):
    def __init__(self, input_channels, num_classes, base_filters=64, use_bn=False):
        super().__init__(input_channels, num_classes, base_filters, use_bn)
        self.model = nn.Sequential(
            self.conv_layer(input_channels, base_filters),
            nn.MaxPool2d(2, 2),
            self.conv_layer(base_filters, base_filters * 2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(base_filters * 2 * 23 * 31, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)

class ResNet(ConfigurableCNN):
    def __init__(self, input_channels, num_classes, base_filters=64, use_bn=False):
        super().__init__(input_channels, num_classes, base_filters, use_bn)
        self.layer1 = ResNetBlock(input_channels, base_filters, use_bn)
        self.layer2 = ResNetBlock(base_filters, base_filters * 2, use_bn)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_filters * 2, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class VGGlike(ConfigurableCNN):
    def __init__(self, input_channels, num_classes, base_filters=64, use_bn=False):
        super().__init__(input_channels, num_classes, base_filters, use_bn)
        self.model = nn.Sequential(
            self.conv_layer(input_channels, base_filters),
            self.conv_layer(base_filters, base_filters),
            nn.MaxPool2d(2, 2),
            self.conv_layer(base_filters, base_filters * 2),
            self.conv_layer(base_filters * 2, base_filters * 2),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(base_filters * 2 * 23 * 31, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


def build_model(cfg: Config) -> nn.Module:
    if cfg.model.model_architecture.TYPE == "ClassicCNN":
        return ClassicCNN(cfg.data.in_channels, cfg.data.num_classes, cfg.model.model_architecture.base_dim)
    elif cfg.model.model_architecture.TYPE == "ResNet":
        return ResNet(cfg.data.in_channels, cfg.data.num_classes, cfg.model.model_architecture.base_dim)
    elif cfg.model.model_architecture.TYPE == "VGGlike":
        return VGGlike(cfg.data.in_channels, cfg.data.num_classes, cfg.model.model_architecture.base_dim)
    else:
        return ClassicCNN(cfg.data.in_channels, cfg.data.num_classes, cfg.model.model_architecture.base_dim)

