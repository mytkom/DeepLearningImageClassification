import torch
import torch.nn as nn
import timm

from configs import Config

# CNNs
class ConfigurableCNN(nn.Module):
    def __init__(
        self,
        input_channels,
        num_classes,
        base_filters=64,
        image_size=32,
        use_bn=False,
        dropout=0.0,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.base_filters = base_filters
        self.use_bn = use_bn
        self.dropout = dropout

    def conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=not self.use_bn,
            )
        ]
        if self.use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout2d(self.dropout))
        return nn.Sequential(*layers)


class ClassicCNN(ConfigurableCNN):
    def __init__(
        self,
        input_channels,
        num_classes,
        base_filters=64,
        image_size=32,
        use_bn=False,
        dropout=0.0,
    ):
        super().__init__(input_channels, num_classes, base_filters, use_bn, dropout)
        self.model = nn.Sequential(
            self.conv_layer(input_channels, base_filters, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            self.conv_layer(base_filters, base_filters * 2),
            nn.MaxPool2d(2, 2),
            self.conv_layer(base_filters * 2, base_filters * 4),
            nn.MaxPool2d(2, 2),
            self.conv_layer(base_filters * 4, base_filters * 8),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(base_filters * 8 * (image_size // 16) * (image_size // 16), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False, dropout=.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        return self.relu(out)


class ResNet(ConfigurableCNN):
    def __init__(
        self,
        input_channels,
        num_classes,
        base_filters=64,
        image_size=32,
        use_bn=False,
        dropout=0.0,
    ):
        super().__init__(input_channels, num_classes, base_filters, use_bn, dropout)
        self.layer1 = ResNetBlock(input_channels, base_filters, use_bn, dropout=self.dropout)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.layer2 = ResNetBlock(base_filters, base_filters * 2, use_bn, dropout=self.dropout)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.layer3 = ResNetBlock(base_filters * 2, base_filters * 4, use_bn, dropout=self.dropout)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.layer4 = ResNetBlock(base_filters * 4, base_filters * 8, use_bn, dropout=self.dropout)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(
            base_filters * 8 * (image_size // 16) * (image_size // 16), 256
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool3(self.layer3(x))
        x = self.pool4(self.layer4(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc3(self.relu(self.fc2(x)))


class VGGlike(ConfigurableCNN):
    def __init__(
        self,
        input_channels,
        num_classes,
        base_filters=64,
        image_size=32,
        use_bn=False,
        dropout=0.0,
    ):
        super().__init__(input_channels, num_classes, base_filters, use_bn, dropout)
        self.model = nn.Sequential(
            self.conv_layer(input_channels, base_filters),
            self.conv_layer(base_filters, base_filters),
            nn.MaxPool2d(2, 2),
            self.conv_layer(base_filters, base_filters * 2),
            self.conv_layer(base_filters * 2, base_filters * 2),
            nn.MaxPool2d(2, 2),
            self.conv_layer(base_filters * 2, base_filters * 4),
            self.conv_layer(base_filters * 4, base_filters * 4),
            self.conv_layer(base_filters * 4, base_filters * 4),
            nn.MaxPool2d(2, 2),
            self.conv_layer(base_filters * 4, base_filters * 8),
            self.conv_layer(base_filters * 8, base_filters * 8),
            self.conv_layer(base_filters * 8, base_filters * 8),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(base_filters * 8 * (image_size // 16) * (image_size // 16), 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.model(x)


CNN_MAP = {"Classic": ClassicCNN, "ResNet": ResNet, "VGGlike": VGGlike}


# parameters based on best performing setup from Appendix A of https://arxiv.org/abs/2210.07240
def ViT3M(in_channels: int, num_classes: int) -> nn.Module:
    return timm.models.VisionTransformer(
        num_classes=num_classes,
        in_chans=in_channels,
        img_size=32,
        patch_size=4,
        embed_dim=192,
        depth=9,
        num_heads=12,
        mlp_ratio=2,
    )


def build_model(cfg: Config) -> nn.Module:
    if cfg.model.architecture == "CNN":
        if cfg.cnn.architecture not in CNN_MAP.keys():
            raise RuntimeError(
                f"Wrong CNN architecture specified: {cfg.cnn.architecture}"
            )
        else:
            return CNN_MAP[cfg.cnn.architecture](
                cfg.data.in_channels,
                cfg.data.num_classes,
                cfg.cnn.base_dim,
                image_size=cfg.data.image_size,
                use_bn=cfg.cnn.batch_normalization,
                dropout=cfg.cnn.dropout,
            )
    else:
        raise RuntimeError(f"Wrong model architecture set: {cfg.model.architecture}")
