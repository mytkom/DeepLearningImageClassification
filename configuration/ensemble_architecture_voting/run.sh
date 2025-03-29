#!/usr/bin/bash

SUBDIR=configuration/ensemble_architecture_voting/
for i in $(seq 1 3);
do
    python3 main.py  --config $SUBDIR/Ensemble.json --ensemble.voting=soft --wandb.name='Ensemble (ViT3M + ResNet3M + soft voting)'
    python3 main.py  --config $SUBDIR/Ensemble.json --ensemble.voting=hard --wandb.name='Ensemble (ViT3M + ResNet3M + hard voting)'
    python3 main.py  --config $SUBDIR/Ensemble.json --ensemble.voting=stacking --wandb.name='Ensemble (ViT3M + ResNet3M + stacking voting)'
done
