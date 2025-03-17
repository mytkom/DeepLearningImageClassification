#!/usr/bin/bash

python3 sweep.py --config configuration/cnn_sweeps/random_wide_ranges/Classic.json
python3 sweep.py --config configuration/cnn_sweeps/random_wide_ranges/ResNet.json
python3 sweep.py --config configuration/cnn_sweeps/random_wide_ranges/VGGlike.json