#!/usr/bin/bash

python3 sweep.py --config configuration/cnn_sweeps/Classic.json
python3 sweep.py --config configuration/cnn_sweeps/ResNet.json
python3 sweep.py --config configuration/cnn_sweeps/VGGlike.json