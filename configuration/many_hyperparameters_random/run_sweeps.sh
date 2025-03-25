#!/usr/bin/bash

SWEEP_SUBDIR=many_hyperparameters_random
python3 sweep.py --config configuration/$SWEEP_SUBDIR/config.json --model.architecture=CNN --cnn.architecture=Classic --project-dir=${SWEEP_SUBDIR}_Classic --sweep.name=Classic
python3 sweep.py --config configuration/$SWEEP_SUBDIR/config.json --model.architecture=CNN --cnn.architecture=ResNet --project-dir=${SWEEP_SUBDIR}_ResNet --sweep.name=ResNet
python3 sweep.py --config configuration/$SWEEP_SUBDIR/config.json --model.architecture=CNN --cnn.architecture=VGGlike --project-dir=${SWEEP_SUBDIR}_VGGlike --sweep.name=VGGlike