#!/usr/bin/bash

python3 sweep.py --config configuration/regularization/config.json --model.architecture=CNN --cnn.architecture=Classic --project-dir=data_augmentation_Classic
python3 sweep.py --config configuration/regularization/config.json --model.architecture=CNN --cnn.architecture=ResNet --project-dir=data_augmentation_ResNet
python3 sweep.py --config configuration/regularization/config.json --model.architecture=CNN --cnn.architecture=VGGlike --project-dir=data_augmentation_VGGlike
# python3 sweep.py --config configuration/regularization/config.json --model.architecture=ViT --project-dir=data_augmentation_ViT