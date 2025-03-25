#!/usr/bin/bash

SWEEP_SUBDIR=data_augmentation
# python3 sweep.py --config configuration/$SWEEP_SUBDIR/config.json --model.architecture=CNN --cnn.architecture=Classic --project-dir=${SWEEP_SUBDIR}_Classic --sweep.name=Classic_extended
python3 sweep.py --config configuration/$SWEEP_SUBDIR/config.json --model.architecture=CNN --cnn.architecture=ResNet --project-dir=${SWEEP_SUBDIR}_ResNet --sweep.name=ResNet
# python3 sweep.py --config configuration/$SWEEP_SUBDIR/config.json --model.architecture=CNN --cnn.architecture=VGGlike --project-dir=${SWEEP_SUBDIR}_VGGlike --sweep.name=VGGlike
# python3 sweep.py --config configuration/$SWEEP_SUBDIR/config.json --model.architecture=ViT --project-dir=${SWEEP_SUBDIR}_ViT