# AMHNN: Adaptive Multi-Scale Hypergraph Neural Network for Multi-Source Detection
This repository contains the official PyTorch implementation for the paper: **"Adaptive Multi-Scale Hypergraph Neural Network for Multi-Source Detection"**.
## 🧩 Repository Structure

* `main.py`: The entry point for generating snapshots and training/evaluating the model.
* `model.py`: Contains the core implementation of the AMHNN framework, including the `AdaptiveMultiScaleSpreadingLayer` and `LocalizedHyperedgeAttention`.
* `data.py`: Handles hypergraph Laplacian computation, topological feature generation, and the multi-source spreading dynamic models (SI, IC, LT).
* `training.py`: Contains the training loop, validation logic, and the dynamic weighted binary cross-entropy loss function.
* `utils.py`: Includes evaluation metrics (Accuracy, F1, AUC) and hypergraph structure augmentation strategies (Dropout and Perturbation).
* `data/`: Directory to store the synthetic and real-world hypergraph datasets.

## ⚙️ Dependencies

Ensure you have the following dependencies installed:

* Python 3.8+
* PyTorch 1.12+
* [dhg](https://github.com/iMoonLab/DeepHypergraph) (Deep Hypergraph)
* numpy
* scipy
* scikit-learn

You can install the required packages using:
```bash
pip install torch dhg numpy scipy scikit-learn
