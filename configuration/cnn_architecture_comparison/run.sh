#!/usr/bin/bash

SUBDIR=configuration/cnn_architecture_comparison/
python3 main.py  --config $SUBDIR/VGGlike.json
python3 main.py  --config $SUBDIR/ResNet.json
python3 main.py  --config $SUBDIR/Classic.json
python3 main.py  --config $SUBDIR/ResNetDeep.json