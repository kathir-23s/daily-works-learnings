import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# --------------------------
# 0) Repro + Device
# --------------------------
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# 1) Load MNIST and build (784, N) + one-hot (10, N)
#    Note: ToTensor() -> [0,1] scale. No mean/std normalization to match your pipeline.
# --------------------------
transform = transforms.Compose([transforms.ToTensor()])

train_raw = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_raw  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Flatten to (N, 784)
X_all = train_raw.data.float().view(-1, 28*28) / 255.0  # (60000, 784)
y_all = train_raw.targets.long()                        # (60000,)

X_test = test_raw.data.float().view(-1, 28*28) / 255.0  # (10000, 784)
y_test = test_raw.targets.long()                        # (10000,)

# Split: 54k train / 6k val
X_train = X_all[:54000].T.contiguous()  # (784, 54000)
y_train_idx = y_all[:54000]             # (54000,)
X_val = X_all[54000:].T.contiguous()    # (784, 6000)
y_val_idx = y_all[54000:]               # (6000,)

# One-hot (10, N)
def to_one_hot(y_idx, num_classes=10):
    N = y_idx.numel()
    oh = torch.zeros(num_classes, N, dtype=torch.float32)
    oh.scatter_(0, y_idx.view(1, N), 1.0)
    return oh

Y_train = to_one_hot(y_train_idx)   # (10, 54000)
Y_val   = to_one_hot(y_val_idx)     # (10, 6000)
Y_test  = to_one_hot(y_test)        # (10, 10000)

# Move to device
X_train, Y_train = X_train.to(device), Y_train.to(device)
X_val,   Y_val   = X_val.to(device),   Y_val.to(device)
X_test,  Y_test  = X_test.to(device),  Y_test.to(device)
y_train_idx = y_train_idx.to(device)
y_val_idx   = y_val_idx.to(device)
y_test      = y_test.to(device)

# --------------------------
# 2) Model that accepts (784, N) and returns (10, N)
# --------------------------
class MLPCols(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward_cols(self, x_cols):
        # x_cols: (784, N)
        x = x_cols.T  # (N, 784) batch-first for nn.Linear
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # logits (N, 10)
        return x.T       # (10, N) to mirror your pipeline

model = MLPCols().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --------------------------
# 3) One-hot cross-entropy (matches categorical CE with one-hot targets)
#    logits_cols: (10, N), targets_onehot_cols: (10, N)
# --------------------------
def onehot_cross_entropy(logits_cols, targets_onehot_cols):
    # compute along columns
    log_probs = F.log_softmax(logits_cols.T, dim=1)  # (N, 10)
    loss = -(targets_onehot_cols.T * log_probs).sum(dim=1).mean()
    return loss

# --------------------------
# 4) Accuracy helper (inputs as columns)
# --------------------------
@torch.no_grad()
def accuracy_cols(logits_cols, y_idx):
    preds = logits_cols.argmax(dim=0)   # (N,)
    return (preds == y_idx).float().mean().item()

# --------------------------
# 5) Training (10 epochs) with column-wise mini-batches
#    Batches are slices over the columns to preserve (784, N) layout
# --------------------------
def train_epoch(batch_size=128):
    model.train()
    N = X_train.shape[1]
    perm = torch.randperm(N, device=device)
    total_loss = 0.0
    steps = 0

    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        xb = X_train[:, idx]      # (784, B)
        yb = Y_train[:, idx]      # (10, B)

        logits = model.forward_cols(xb)  # (10, B)
        loss = onehot_cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / steps

@torch.no_grad()
def evaluate(X_cols, y_idx, batch_size=512):
    model.eval()
    N = X_cols.shape[1]
    total_acc = 0.0
    steps = 0
    for i in range(0, N, batch_size):
        idx = slice(i, min(i+batch_size, N))
        logits = model.forward_cols(X_cols[:, idx])  # (10, b)
        total_acc += accuracy_cols(logits, y_idx[i:i+logits.shape[1]])
        steps += 1
    return total_acc / steps

# --------------------------
# 6) Run 10 epochs
# --------------------------
for epoch in range(1, 11):
    train_loss = train_epoch(batch_size=128)
    val_acc = evaluate(X_val, y_val_idx, batch_size=1024)
    print(f"Epoch {epoch:02d} | train_loss: {train_loss:.4f} | val_acc: {val_acc*100:.2f}%")

# --------------------------
# 7) Final test accuracy
# --------------------------
test_acc = evaluate(X_test.T.contiguous().T, y_test, batch_size=1024)  # X_test already (N,784); ensure (784,N)
# Simpler: build X_test_cols directly
X_test_cols = X_test.T.contiguous().to(device)  # (784, 10000)
test_acc = evaluate(X_test_cols, y_test, batch_size=1024)
print(f"Test Accuracy: {test_acc*100:.2f}%")
