# test2_mnist.py
import os
import time
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # use non-GUI backend to avoid blocking windows
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from models.mlp import MLP
from tensor.tensor_scratch import TensorT

# -----------------------------
# 0) Utilities
# -----------------------------
original_init = TensorT.__init__

def patched_init(self, data, _op=None, _parent=()):
    # keep parity with test1.py patch
    if isinstance(data, TensorT):
        data = data.data
    elif isinstance(data, (float, int)):
        data = [[float(data)]]
    elif isinstance(data, list) and data and not isinstance(data[0], list):
        data = [data]
    original_init(self, data, _op=_op, _parent=_parent)

TensorT.__init__ = patched_init

def one_hot(y, num_classes=10):
    """y: (N,) ints -> (num_classes, N) one-hot"""
    N = y.shape[0]
    oh = np.zeros((num_classes, N), dtype=np.float32)
    oh[y, np.arange(N)] = 1.0
    return oh

def show_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    ticks = np.arange(10)
    ax.set_xticks(ticks); ax.set_xticklabels(ticks)
    ax.set_yticks(ticks); ax.set_yticklabels(ticks)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # IMPORTANT: free the figure


# -----------------------------
# 1) Load MNIST (robust)
# -----------------------------
# Priority 1: local numpy file (fast, offline). Put `mnist.npz` in ./data if you have it.
#   You can get it once via: from tensorflow.keras.datasets import mnist; mnist.load_data()
#   It saves at ~/.keras/datasets/mnist.npz — copy it to ./data/mnist.npz to avoid net.
local_npz = "data/mnist.npz"
if os.path.exists(local_npz):
    with np.load(local_npz) as d:
        X_train_np = d["x_train"]
        y_train_np = d["y_train"]
        X_test_np  = d["x_test"]
        y_test_np  = d["y_test"]
else:
    # Priority 2: scikit-learn OpenML (needs internet). If that fails, raise with a clear msg.
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False)
        X_all = mnist.data.reshape(-1, 28, 28).astype(np.uint8)
        y_all = mnist.target.astype(np.int64)
        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            X_all, y_all, test_size=10000, random_state=42, stratify=y_all
        )
    except Exception as e:
        raise RuntimeError(
            "No local MNIST and OpenML fetch failed. "
            "Place mnist.npz at ./data/mnist.npz or ensure internet."
        ) from e

print(f"[DEBUG] Raw MNIST shapes: "
      f"train {X_train_np.shape}, test {X_test_np.shape}")

# -----------------------------
# 2) Preprocess -> float32 [0,1], flatten, split val, TensorT
# -----------------------------
X_train_np = (X_train_np.astype(np.float32) / 255.0).reshape(-1, 28*28)   # (N, 784)
X_test_np  = (X_test_np.astype(np.float32)  / 255.0).reshape(-1, 28*28)   # (N, 784)

# Optional validation split (10% of train)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_np, y_train_np, test_size=0.1, random_state=42, stratify=y_train_np
)

# Transpose to match your MLP convention: (features, samples)
X_tr_T  = X_tr.T                      # (784, N_tr)
X_val_T = X_val.T                     # (784, N_val)
X_te_T  = X_test_np.T                 # (784, N_te)

Y_tr_oh  = one_hot(y_tr, 10)          # (10, N_tr)
Y_val_oh = one_hot(y_val, 10)         # (10, N_val)

# Convert to TensorT
X_train = TensorT(X_tr_T.tolist())
Y_train = TensorT(Y_tr_oh.tolist())
X_val_t = TensorT(X_val_T.tolist())
Y_val_t = TensorT(Y_val_oh.tolist())
X_test  = TensorT(X_te_T.tolist())

print(f"[DEBUG] Tensor shapes: X_train {X_train.shape}, Y_train {Y_train.shape}, "
      f"X_val {X_val_t.shape}, Y_val {Y_val_t.shape}, X_test {X_test.shape}")

# -----------------------------
# 3) Build MLP (classification)
# -----------------------------
# Notes:
#  - output_size=10 (digits 0..9)
#  - use 'relu' hidden; final layer should be linear -> softmax inside loss (if supported).
#  - if your MLP doesn't implement cross-entropy+softmax, set loss to MSE with one-hot.
LOSS = 'cross_entropy_loss'  # fallback to 'mean_squared_error' if CE not implemented

mlp = MLP(
    input_size=784,
    hidden_layers=[128, 64],
    output_size=10,
    weight_initialization='he_normal',
    activation_func='relu',
    loss_function=LOSS,
    learning_rate=0.01
)

# -----------------------------
# 4) Training loop (full-batch like your test1)
# -----------------------------
def train_debug(self, X, Y, X_val=None, Y_val=None, epochs=10, print_every=1):
    print(f"\n--- Starting MNIST Training for {epochs} epochs ---")
    t0 = time.time()
    result = []

    for ep in range(1, epochs+1):
        # Forward + loss
        AL = self.forward(X)           # (10, N)
        loss = self.cost(Y, AL)        # scalar TensorT

        # Backprop
        loss.zero_grad()
        loss.backward()

        # --- DEBUG: last-layer gradient sanity (only first 3 epochs) ---
        if ep <= 3 and hasattr(self, 'debug_check_last_grad'):
            self.debug_check_last_grad(Y)

        # --- DEBUG: mean |grad| per layer BEFORE update ---
        for l, (W, B) in enumerate(zip(self.weights, self.biases), start=1):
            gW = 0.0; nW = 0
            if W.grad:
                for row in W.grad:
                    for v in row:
                        gW += abs(v); nW += 1
            gB = 0.0; nB = 0
            if B.grad:
                for row in B.grad:
                    for v in row:
                        gB += abs(v); nB += 1
            print(f"[grad] L{l}  |W|_mean={gW/max(1,nW):.3e}  |b|_mean={gB/max(1,nB):.3e}")

        # --- stash params, then update ---
        prev_W = [[row[:] for row in W.data] for W in self.weights]
        prev_b = [[row[:] for row in b.data] for b in self.biases]

        self.update_parameters()

        # --- DEBUG: how much params changed in this step ---
        for l, (W, B) in enumerate(zip(self.weights, self.biases), start=1):
            dW = 0.0
            for i in range(len(W.data)):
                for j in range(len(W.data[0])):
                    dW += abs(W.data[i][j] - prev_W[l-1][i][j])
            dB = 0.0
            for i in range(len(B.data)):
                dB += abs(B.data[i][0] - prev_b[l-1][i][0])
            print(f"[step] L{l}  ΔW_sum={dW:.3e}  Δb_sum={dB:.3e}")

        # Log once per epoch
        if (ep % print_every) == 0 or ep == epochs:
            tr_loss = float(loss.data[0][0])
            msg = f"Epoch {ep:>3} | train_loss: {tr_loss:.4f}"
            if X_val is not None and Y_val is not None:
                logits = self.forward(X_val)        # (10, N_val)
                pred_val = np.argmax(np.array(logits.data), axis=0)
                true_val = np.argmax(np.array(Y_val.data), axis=0)
                val_acc = (pred_val == true_val).mean()
                msg += f" | val_acc: {val_acc*100:.2f}%"
            print(msg)
            result.append(tr_loss)

    print(f"--- Finished in {time.time()-t0:.2f}s ---")
    return result



# Bind and train
costs = train_debug(mlp, X_train, Y_train, X_val_t, Y_val_t, epochs=10, print_every=1)

# Save weights
mlp.save_weights("mnist_mlp_weights.pkl")

# -----------------------------
# 5) Evaluation on test set
# -----------------------------
# Forward -> predictions
logits_test = mlp.predict(X_test)      # (10, N_te) or (N_te, 10) depending on your MLP
probs = np.array(logits_test.data)
if probs.shape[0] != 10 and probs.shape[1] == 10:
    probs = probs.T
y_pred = np.argmax(probs, axis=0)

# true labels
y_true = y_test_np

# Metrics
acc = (y_pred == y_true).mean()
print(f"\n=== Test Metrics ===")
print(f"Accuracy: {acc*100:.2f}%")
print(classification_report(y_true, y_pred, digits=4))

# Confusion matrix
show_confusion_matrix(y_true, y_pred, title="MNIST Confusion Matrix", save_path="confusion_matrix.png")

# -----------------------------
# 6) Quick sanity predictions
# -----------------------------
def preview_samples(X_flat, y_true, y_pred, count=8, save_path="predictions_preview.png"):
    idx = np.random.choice(X_flat.shape[0], count, replace=False)
    fig, axes = plt.subplots(1, count, figsize=(1.2*count, 2.6))
    if count == 1:
        axes = [axes]
    for i, k in enumerate(idx):
        ax = axes[i]
        ax.imshow(X_flat[k].reshape(28, 28), cmap='gray')
        ax.axis('off')
        ax.set_title(f"T:{y_true[k]} P:{y_pred[k]}")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # IMPORTANT

preview_samples(X_test_np, y_true, y_pred, count=10, save_path="predictions_preview.png")
print("Saved plots: confusion_matrix.png, predictions_preview.png")

# -----------------------------
# 7) Weight reload smoke-test
# -----------------------------
mlp2 = MLP(
    input_size=784, hidden_layers=[256,128,64], output_size=10,
    weight_initialization='he_normal', activation_func='relu',
    loss_function=LOSS, learning_rate=0.01
)
mlp2.load_weights("mnist_mlp_weights_10.pkl")
logits_test2 = mlp2.predict(X_test)
probs2 = np.array(logits_test2.data)
if probs2.shape[0] != 10 and probs2.shape[1] == 10:
    probs2 = probs2.T
y_pred2 = np.argmax(probs2, axis=0)
acc2 = (y_pred2 == y_true).mean()
print(f"[DEBUG] Reloaded-model test accuracy: {acc2*100:.2f}%")
