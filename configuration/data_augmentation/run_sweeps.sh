#!/usr/bin/bash

python3 sweep.py --config configuration/data_augmentation/config.json --model.architecture=CNN --cnn.architecture=Classic --project_dir=data_augmentation_Classic
python3 sweep.py --config configuration/data_augmentation/config.json --model.architecture=CNN --cnn.architecture=ResNet --project_dir=data_augmentation_ResNet
python3 sweep.py --config configuration/data_augmentation/config.json --model.architecture=CNN --cnn.architecture=VGGlike --project_dir=data_augmentation_VGGlike
python3 sweep.py --config configuration/data_augmentation/config.json --model.architecture=ViT --project_dir=data_augmentation_ViT