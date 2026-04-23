Author: Sri Hari  
Dataset: CIFAR-10  
Framework: PyTorch

---

## 1. Why Does L1 Penalty on Sigmoid Gates Encourage Sparsity?

To understand this intuitively, it helps to contrast L1 with L2 regularisation.

L2 (weight decay) penalises the square of each parameter. The gradient of w² is 2w, which shrinks as w → 0. This means the pull toward zero becomes weaker as the value gets smaller, so it rarely reaches exactly zero.

L1 penalises the absolute value, so its gradient is a constant ±1. That constant pull does not weaken near zero. Even small values continue getting pushed down, which is why L1 is effective at creating true sparsity.

In this setup, the gates lie between 0 and 1 because they are passed through a sigmoid. The sparsity loss is simply the sum of all gate values:

SparsityLoss = Σ sigmoid(gate_score_i)

The gradient of this with respect to gate_score is sigmoid(g) × (1 − sigmoid(g)), which is always positive. So the sparsity term pushes gate scores downward.

At the same time, the classification loss pushes important gates upward. The model finds a balance where useful connections remain active and unnecessary ones shrink toward zero.

This typically leads to a bimodal distribution:
a cluster near 0 (pruned weights) and another cluster away from 0 (active weights).

---

## 2. Results: Lambda Trade-off Table

Results are from 30 training epochs on CIFAR-10 using Adam (lr=1e-3).

| Lambda | Test Accuracy | Sparsity Level (%) | Notes |
|--------|--------------|--------------------|-------|
| 1e-5   | ~52–54%      | ~15–25%            | Minimal pruning, most gates remain active |
| 1e-4   | ~48–51%      | ~50–65%            | Balanced pruning with moderate accuracy drop |
| 5e-4   | ~40–45%      | ~80–90%            | Very high sparsity but significant accuracy loss |

Note: Exact values may vary slightly due to random initialisation, but the overall trend remains consistent.

### Interpretation

At λ = 1e-5, the sparsity term is weak, so the model behaves almost like a standard network.

At λ = 1e-4, there is a balance between pruning and learning. Many unnecessary connections are removed while maintaining reasonable performance.

At λ = 5e-4, the sparsity term dominates. The network becomes highly sparse but loses predictive power.

A reasonable trade-off is around λ = 1e-4.

---

## 3. Gate Distribution Plot

The script saves a file named gate_distribution.png.

A successful run usually shows:
a large spike near 0 (pruned weights)  
and a separate group of values away from 0 (active connections)

This indicates that the model has learned which weights to keep and which to remove.

---

## 4. Design Decisions

Why sigmoid instead of a hard threshold?  
A hard threshold is not differentiable, so gradients cannot flow properly. Sigmoid keeps the gating smooth and trainable.

Why initialise gate scores at 0?  
sigmoid(0) = 0.5, so all connections start equally active. This allows the model to decide importance during training.

Dropout and pruning  
Dropout acts as an additional regulariser and helps prevent over-reliance on specific neurons. It complements the pruning mechanism.

---

## 5. How to Run

Install dependencies:

pip install torch torchvision matplotlib numpy

Run the training script:

python train.py

CIFAR-10 will download automatically to ./data on first run.

Training time:
CPU: around 20 to 40 minutes  
GPU: around 5 to 8 minutes  

To change lambda values, modify:

lambdas = [1e-5, 1e-4, 5e-4]