import torch
import torch.nn as nn
import timm

from configs import Config

# CNNs
import torch
import torch.nn as nn


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
        self.image_size = image_size

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
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def fc_layers(self, in_features):
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(min(self.dropout * 1.5, 1.0)),
            nn.Linear(256, self.num_classes),
        )


class ClassicCNN(ConfigurableCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = nn.Sequential(
            self.conv_layer(self.input_channels, self.base_filters, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            self.conv_layer(self.base_filters, self.base_filters * 2),
            nn.MaxPool2d(2, 2),
            self.conv_layer(self.base_filters * 2, self.base_filters * 4),
            nn.MaxPool2d(2, 2),
            self.conv_layer(self.base_filters * 4, self.base_filters * 8),
            nn.MaxPool2d(2, 2),
        )
        in_features = self.base_filters * 8 * (self.image_size // 16) ** 2
        self.fc = self.fc_layers(in_features)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=not use_bn
        )
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=not use_bn
        )
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU()
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False, stride=stride)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet(ConfigurableCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = nn.Sequential(
            ResNetBlock(self.input_channels, self.base_filters, self.use_bn),
            nn.MaxPool2d(2, 2),
            ResNetBlock(self.base_filters, self.base_filters * 2, self.use_bn),
            nn.MaxPool2d(2, 2),
            ResNetBlock(self.base_filters * 2, self.base_filters * 4, self.use_bn),
            nn.MaxPool2d(2, 2),
            ResNetBlock(self.base_filters * 4, self.base_filters * 8, self.use_bn),
            nn.MaxPool2d(2, 2),
        )
        in_features = self.base_filters * 8 * (self.image_size // 16) ** 2
        self.fc = self.fc_layers(in_features)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# after a while, it turns out that in comparison to original ResNets our
# architectures have little number of layers https://arxiv.org/abs/1512.03385
# we would check if ResNetDeep it performs better that ResNet class
class ResNetDeep(ConfigurableCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = nn.Sequential(
            ResNetBlock(self.input_channels, self.base_filters, self.use_bn),
            nn.MaxPool2d(2, 2),
            # 3 res layers
            ResNetBlock(self.base_filters, self.base_filters, self.use_bn),
            ResNetBlock(self.base_filters, self.base_filters, self.use_bn),
            ResNetBlock(self.base_filters, self.base_filters, self.use_bn),
            # stride, 4 res layers
            ResNetBlock(self.base_filters, self.base_filters * 2, self.use_bn, stride=2),
            ResNetBlock(self.base_filters * 2, self.base_filters * 2, self.use_bn),
            ResNetBlock(self.base_filters * 2, self.base_filters * 2, self.use_bn),
            ResNetBlock(self.base_filters * 2, self.base_filters * 2, self.use_bn),
            # stride, 6 res layers
            ResNetBlock(self.base_filters * 2, self.base_filters * 4, self.use_bn, stride=2),
            ResNetBlock(self.base_filters * 4, self.base_filters * 4, self.use_bn),
            ResNetBlock(self.base_filters * 4, self.base_filters * 4, self.use_bn),
            ResNetBlock(self.base_filters * 4, self.base_filters * 4, self.use_bn),
            ResNetBlock(self.base_filters * 4, self.base_filters * 4, self.use_bn),
            ResNetBlock(self.base_filters * 4, self.base_filters * 4, self.use_bn),
            # stride, 3 res layers
            ResNetBlock(self.base_filters * 4, self.base_filters * 8, self.use_bn, stride=2),
            ResNetBlock(self.base_filters * 8, self.base_filters * 8, self.use_bn),
            ResNetBlock(self.base_filters * 8, self.base_filters * 8, self.use_bn),
        )
        in_features = self.base_filters * 8 * (self.image_size // 16) ** 2
        self.fc = self.fc_layers(in_features)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class VGGlike(ConfigurableCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = nn.Sequential(
            self.conv_layer(self.input_channels, self.base_filters),
            self.conv_layer(self.base_filters, self.base_filters),
            nn.MaxPool2d(2, 2),
            self.conv_layer(self.base_filters, self.base_filters * 2),
            self.conv_layer(self.base_filters * 2, self.base_filters * 2),
            nn.MaxPool2d(2, 2),
            self.conv_layer(self.base_filters * 2, self.base_filters * 4),
            self.conv_layer(self.base_filters * 4, self.base_filters * 4),
            self.conv_layer(self.base_filters * 4, self.base_filters * 4),
            nn.MaxPool2d(2, 2),
            self.conv_layer(self.base_filters * 4, self.base_filters * 8),
            self.conv_layer(self.base_filters * 8, self.base_filters * 8),
            self.conv_layer(self.base_filters * 8, self.base_filters * 8),
            nn.MaxPool2d(2, 2),
        )
        in_features = self.base_filters * 8 * (self.image_size // 16) ** 2
        self.fc = self.fc_layers(in_features)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


CNN_MAP = {"Classic": ClassicCNN, "ResNet": ResNet, "ResNetDeep": ResNetDeep, "VGGlike": VGGlike}


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
    print("config just before model creation: ", cfg)
    if cfg.model.architecture == "CNN":
        if cfg.cnn.architecture not in CNN_MAP.keys():
            raise RuntimeError(f"Wrong CNN architecture specified: {cfg.cnn.architecture}")
        else:
            return CNN_MAP[cfg.cnn.architecture](
                cfg.data.in_channels,
                cfg.data.num_classes,
                cfg.cnn.base_dim,
                image_size=cfg.data.image_size,
                use_bn=cfg.cnn.batch_normalization,
                dropout=cfg.cnn.dropout,
            )
    elif cfg.model.architecture == "ViT":
        return ViT3M(cfg.data.in_channels, cfg.data.num_classes)
    else:
        raise RuntimeError(f"Wrong model architecture set: {cfg.model.architecture}")
