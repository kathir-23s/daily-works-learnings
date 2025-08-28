# predict_100.py
import os, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensor.tensor_scratch import TensorT
from models.mlp import MLP

# -----------------------------
# 1) Load MNIST (local npz preferred)
# -----------------------------
local_npz = "data/mnist.npz"
if os.path.exists(local_npz):
    with np.load(local_npz) as d:
        X_train_np = d["x_train"]
        y_train_np = d["y_train"]
        X_test_np  = d["x_test"]
        y_test_np  = d["y_test"]
else:
    # Priority 2: fetch via OpenML (needs internet)
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml("mnist_784", version=1, as_frame=False)
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


print(f"[DEBUG] Raw MNIST shapes: train {X_train_np.shape}, test {X_test_np.shape}")


# Flatten + normalize
X_test_flat = X_test_np.reshape(X_test_np.shape[0], -1) / 255.0
y_test_raw  = y_test_np


# -----------------------------
# 2) Reload trained MLP
# -----------------------------
mlp = MLP(
    input_size=784,
    hidden_layers=[128, 64],
    output_size=10,        # same architecture used in training
    activation_func="relu",
    loss_function="cross_entropy_loss",
    weight_initialization="he_normal",
    learning_rate=0.01
)

mlp.load_weights("mlp_weights_2.pkl") 
# -----------------------------
# 3) Run predictions
# -----------------------------
X_test = TensorT(X_test_flat.T.tolist())
logits = mlp.forward(X_test)  # shape (10, N)
probs  = np.array(logits.data)
y_pred = np.argmax(probs, axis=0)

# -----------------------------
# 4) Visualize 100 predictions with color-coded labels
# -----------------------------
count = 100
idxs = np.random.choice(X_test_flat.shape[0], count, replace=False)

fig, axes = plt.subplots(10, 10, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    img = X_test_flat[idxs[i]].reshape(28, 28)
    true = y_test_raw[idxs[i]]
    pred = y_pred[idxs[i]]

    ax.imshow(img, cmap="gray")
    color = "green" if pred == true else "red"
    ax.set_title(f"T:{true}, P:{pred}", fontsize=7, color=color)
    ax.axis("off")

plt.tight_layout()
plt.savefig("predictions_100_1.png", dpi=150)
plt.close()
print("Saved predictions grid to predictions_100.png (green=correct, red=wrong)")
