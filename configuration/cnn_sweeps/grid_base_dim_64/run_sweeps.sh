#!/usr/bin/bash

python3 sweep.py --config configuration/cnn_sweeps/grid_base_dim_64/Classic.json
python3 sweep.py --config configuration/cnn_sweeps/grid_base_dim_64/ResNet.json
python3 sweep.py --config configuration/cnn_sweeps/grid_base_dim_64/VGGlike.json