#!/usr/bin/bash

python3 sweep.py --config configuration/cnn_sweeps/grid/config.json --cnn.architecture=Classic
# python3 sweep.py --config configuration/cnn_sweeps/grid/config.json --cnn.architecture=ResNet
# python3 sweep.py --config configuration/cnn_sweeps/grid/config.json --cnn.architecture=VGGlike
# python3 sweep.py --config configuration/cnn_sweeps/grid/config.json --cnn.architecture=Classic --cnn.base_dim=128
# python3 sweep.py --config configuration/cnn_sweeps/grid/config.json --cnn.architecture=ResNet --cnn.base_dim=128
# python3 sweep.py --config configuration/cnn_sweeps/grid/config.json --cnn.architecture=VGGlike --cnn.base_dim=128