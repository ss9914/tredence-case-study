# Self Pruning Neural Network (Tredence Case Study)

This project implements a simple feed-forward neural network in PyTorch that learns to prune its own weights during training. Instead of removing weights after training, the model gradually reduces unnecessary connections on its own.

## Overview

The main idea is to associate every weight with a learnable gate value. This gate controls how much the weight contributes during the forward pass.

Each gate is passed through a sigmoid so its value stays between 0 and 1.

Forward pass looks like:

pruned_weight = weight * sigmoid(gate_score)

If a gate becomes very small (close to 0), that weight is effectively removed from the network.

## Loss Function

The training loss has two parts:

Total Loss = CrossEntropy + λ * Sparsity Loss

* CrossEntropy → for classification (CIFAR-10)
* Sparsity Loss → sum of all gate values

The sparsity term encourages the model to reduce the number of active connections.

## Why L1 for Sparsity?

L1 works well here because it pushes values directly toward zero.

Since gate values are always positive after sigmoid, minimizing their sum reduces the number of active weights. This helps the model naturally become sparse.

## Model

The network is a simple MLP using custom linear layers:

* Flatten input (CIFAR-10 images)
* Multiple hidden layers with ReLU
* Final classification layer

Each linear layer is replaced with a custom `PrunableLinear` layer.

## PrunableLinear Layer

This is the key part of the implementation.

Each layer has:

* weights
* bias
* gate parameters (same shape as weights)

During forward pass:

* gates = sigmoid(gate_param)
* masked weights = weight * gates

Gradients flow through both weights and gates, so the model learns which connections to keep.

## Training

* Dataset: CIFAR-10
* Optimizer: Adam
* Batch size: 128
* Loss = classification + sparsity penalty

Different values of λ are used to observe the trade-off between accuracy and sparsity.

## Results (approximate)

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 1e-5   | ~53%     | ~20%     |
| 1e-4   | ~50%     | ~55-60%  |
| 5e-4   | ~40-45%  | ~80%+    |

## Observations

* Smaller λ keeps more weights active, so accuracy is higher
* Larger λ forces more pruning, but hurts accuracy
* There is a clear trade-off between model size and performance

## Output

* Training logs (loss, accuracy)
* Sparsity percentage
* Histogram of gate values (`gate_distribution.png`)

For a good model, most gates cluster near zero with some remaining active.

## How to Run

Install dependencies:

pip install torch torchvision matplotlib numpy

Run:

python train.py

CIFAR-10 will download automatically.

## Notes

The implementation is intentionally kept simple. The focus is on understanding how gating + L1 regularization leads to self-pruning behavior rather than building a highly optimized model.
