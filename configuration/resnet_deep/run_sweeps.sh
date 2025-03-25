#!/usr/bin/bash

SWEEP_SUBDIR=resnet_deep
python3 sweep.py --config configuration/$SWEEP_SUBDIR/config.json --model.architecture=CNN --cnn.architecture=ResNetDeep --project-dir=${SWEEP_SUBDIR}_ResNetDeep --sweep.name=ResNetDeep
python3 sweep.py --config configuration/$SWEEP_SUBDIR/config.json --model.architecture=CNN --cnn.architecture=ResNet --project-dir=${SWEEP_SUBDIR}_ResNet --sweep.name=ResNet