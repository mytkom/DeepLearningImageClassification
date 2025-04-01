# Deep Learning Image Classification Project

## 🚀 Introduction

This project explores various deep learning architectures for image classification tasks, focusing on comparative analysis of their performance. We implemented and evaluated ResNet, VGG-like, and Classic CNN models trained from scratch, as well as experimented with Vision Transformer (ViT) model. For every model we created hyperparameters sweep and then trained models for optimal hyperparameters. This repository provides the code and configurations of our experiments.

## 📂 Folder Structure

```plaintext
📦deep-learning-image-classification
 ├── 📂configs                # Configuration files for experiments
 │   ├── 📄config_utils.py    # Utils for showning or saving configs
 │   └── 📄config.py          # Main configuration script
 ├── 📂configuration          # Experiment-specific configuration files
 │   ├── 📂cnn_architecture_comparison
 │   ├── 📂data_augmentation
 │   ├── 📂few_shot_sweep
 │   ├── 📂ensemble_architecture_voting
 │   ├── 📂lr_and_base_dim
 │   ├── 📂lr_and_bs
 │   ├── 📂many_hyperparameters_random
 │   ├── 📂pretrained_sweeps
 │   ├── 📂regularization
 │   └── 📂resnet_deep
 ├── 📂dataset                # Data loading and preprocessing modules
 │   └── 📄dataset.py         # Data loader and preprocessing scripts
 ├── 📂modeling               # Model architecture definitions
 │   ├── 📄loss.py            # Loss function
 │   └── 📄model.py           # All architecture classes
 ├── 📂utils                  # Utility scripts for various tasks
 │   ├── 📄logger.py          # Logging utilities
 │   └── 📄metrics.py         # Performance metrics
 ├── 📂engine                 # Training and validation engine
 │   ├── 📄base_engine.py     # Base engine class
 │   ├── 📄sweep_engine.py    # Sweep engine class
 │   ├── 📄few_shot_engine.py # Few-shot learning engine class
 │   └── 📄engine.py          # Training and validation loops
 ├── 📄.gitignore             # Specifies intentionally untracked files
 ├── 📄LICENSE                # License file
 ├── 📄README.md              # Project README
 ├── 📄linter.sh              # Code formatting script
 ├── 📄requirements.txt       # Dependencies
 └── 📄main.py                # Main training script
 └── 📄sweep.py               # Sweep training script
```

## ⚙️ Configuration

Experiment configurations are stored in the `configuration` directory. WandB sweeps are heavily utilized for hyperparameter tuning and experiment tracking. Reproducibility is maintained by setting random seeds in configuration files.

## 🏋️‍♂️ Training

### Basic Usage

1.  **Set up the environment:**

    ```shell
    pip install uv
    uv sync
    ```

2.  **Run training scripts:**

    Example for training a Ensemble model:

    ```shell
    python main.py --config configuration/ensemble/Ensemble.json
    ```

3.  **Run WandB sweeps:**

    Example to run a WandB sweep for data augmentation experiments:

    ```shell
    ./configuration/data_augmentation/run_sweeps.sh
    ```

    Ensure you are logged into your WandB account.
