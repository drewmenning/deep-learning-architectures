import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
repo = Path(".")
data_dir = repo / "data"
results_dir = repo / "results"
results_dir.mkdir(exist_ok=True)

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
print("Device:", device)

def plot_history(history, title_prefix):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} Accuracy")
    plt.legend()
    plt.tight_layout()
    return plt.gcf()

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total_count += xb.size(0)

    return total_loss / total_count, total_correct / total_count

@torch.no_grad()
def evaluate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        total_correct += (logits.argmax(dim=1) == yb).sum().item()
        total_count += xb.size(0)

    return total_loss / total_count, total_correct / total_count

def fit_classifier(model, train_loader, val_loader, epochs=10, lr=1e-3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion)

        history["train_loss"].append(round(train_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    return history

@torch.no_grad()
def predict_classifier(model, loader):
    model.eval()
    all_logits = []
    all_targets = []

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).cpu()
        all_logits.append(logits)
        all_targets.append(yb)

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    preds = logits.argmax(dim=1)
    return logits, preds, targets

def save_image_grid(images, labels, preds, title, out_path, max_items=8):
    count = min(max_items, len(images))
    plt.figure(figsize=(12, 6))
    for i in range(count):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"T: {class_names[labels[i]]}\nP: {class_names[preds[i]]}", fontsize=9)
        plt.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()

def load_fashion_subset():
    train_df = pd.read_csv(data_dir / "fashion_train.csv")
    test_df = pd.read_csv(data_dir / "fashion_test.csv")

    X_train_full = train_df.drop(columns=["y"]).to_numpy(dtype=np.float32) / 255.0
    y_train_full = train_df["y"].to_numpy(dtype=np.int64)
    X_test = test_df.drop(columns=["y"]).to_numpy(dtype=np.float32) / 255.0
    y_test = test_df["y"].to_numpy(dtype=np.int64)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train_full
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = load_fashion_subset()
print("Fashion subset shapes:", X_train.shape, X_val.shape, X_test.shape)

# Part A: Multilayer Perceptron (MLP)

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

train_ds_mlp = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
val_ds_mlp = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
test_ds_mlp = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader_mlp = DataLoader(train_ds_mlp, batch_size=64, shuffle=True)
val_loader_mlp = DataLoader(val_ds_mlp, batch_size=128, shuffle=False)
test_loader_mlp = DataLoader(test_ds_mlp, batch_size=128, shuffle=False)

mlp_model = MLPClassifier().to(device)
mlp_history = fit_classifier(mlp_model, train_loader_mlp, val_loader_mlp, epochs=15, lr=1e-3)

criterion = nn.CrossEntropyLoss()
mlp_test_loss, mlp_test_acc = evaluate_epoch(mlp_model, test_loader_mlp, criterion)
print("MLP test loss:", round(mlp_test_loss, 4))
print("MLP test accuracy:", round(mlp_test_acc, 4))

fig = plot_history(mlp_history, "MLP")
fig.savefig(results_dir / "mlp_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

# Part B: Convolutional Neural Network (CNN)

X_train_img = X_train.reshape(-1, 1, 28, 28)
X_val_img = X_val.reshape(-1, 1, 28, 28)
X_test_img = X_test.reshape(-1, 1, 28, 28)

train_ds_cnn = TensorDataset(torch.tensor(X_train_img), torch.tensor(y_train))
val_ds_cnn = TensorDataset(torch.tensor(X_val_img), torch.tensor(y_val))
test_ds_cnn = TensorDataset(torch.tensor(X_test_img), torch.tensor(y_test))

train_loader_cnn = DataLoader(train_ds_cnn, batch_size=64, shuffle=True)
val_loader_cnn = DataLoader(val_ds_cnn, batch_size=128, shuffle=False)
test_loader_cnn = DataLoader(test_ds_cnn, batch_size=128, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

cnn_model = SimpleCNN().to(device)
cnn_history = fit_classifier(cnn_model, train_loader_cnn, val_loader_cnn, epochs=12, lr=1e-3)

cnn_test_loss, cnn_test_acc = evaluate_epoch(cnn_model, test_loader_cnn, criterion)
print("CNN test loss:", round(cnn_test_loss, 4))
print("CNN test accuracy:", round(cnn_test_acc, 4))

fig = plot_history(cnn_history, "CNN")
fig.savefig(results_dir / "cnn_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

cnn_logits, cnn_preds, cnn_targets = predict_classifier(cnn_model, test_loader_cnn)
cm = confusion_matrix(cnn_targets.numpy(), cnn_preds.numpy())

plt.figure(figsize=(8, 7))
plt.imshow(cm, interpolation="nearest")
plt.title("CNN Confusion Matrix")
plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right", fontsize=8)
plt.yticks(range(len(class_names)), class_names, fontsize=8)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()
plt.savefig(results_dir / "cnn_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

correct_idx = torch.where(cnn_preds == cnn_targets)[0][:8].tolist()
incorrect_idx = torch.where(cnn_preds != cnn_targets)[0][:8].tolist()

save_image_grid(
    X_test_img[correct_idx],
    y_test[correct_idx],
    cnn_preds.numpy()[correct_idx],
    "CNN Correct Predictions",
    results_dir / "cnn_correct_examples.png",
)

save_image_grid(
    X_test_img[incorrect_idx],
    y_test[incorrect_idx],
    cnn_preds.numpy()[incorrect_idx],
    "CNN Incorrect Predictions",
    results_dir / "cnn_incorrect_examples.png",
)

# Part C: Recurrent Neural Network / LSTM

text = (data_dir / "tinyshakespeare.txt").read_text(encoding="utf-8")[:250_000]
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

encoded = np.array([stoi[ch] for ch in text], dtype=np.int64)
split_idx = int(len(encoded) * 0.9)
train_encoded = encoded[:split_idx]
val_encoded = encoded[split_idx:]

class CharSeqDataset(Dataset):
    def __init__(self, data, seq_len=60, stride=3):
        self.data = data
        self.seq_len = seq_len
        self.starts = list(range(0, len(data) - seq_len - 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        x = self.data[start:start + self.seq_len]
        y = self.data[start + 1:start + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

seq_len = 60
train_ds_rnn = CharSeqDataset(train_encoded, seq_len=seq_len, stride=3)
val_ds_rnn = CharSeqDataset(val_encoded, seq_len=seq_len, stride=3)
train_loader_rnn = DataLoader(train_ds_rnn, batch_size=64, shuffle=True)
val_loader_rnn = DataLoader(val_ds_rnn, batch_size=64, shuffle=False)

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden

rnn_model = CharLSTM(vocab_size).to(device)
rnn_optimizer = torch.optim.Adam(rnn_model.parameters(), lr=1e-3)
rnn_criterion = nn.CrossEntropyLoss()

def train_rnn_epoch(model, loader):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        rnn_optimizer.zero_grad()
        logits, _ = model(xb)
        loss = rnn_criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        rnn_optimizer.step()
        total_loss += loss.item() * xb.numel()
        total_tokens += xb.numel()
    return total_loss / total_tokens

@torch.no_grad()
def eval_rnn_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits, _ = model(xb)
        loss = rnn_criterion(logits.reshape(-1, vocab_size), yb.reshape(-1))
        total_loss += loss.item() * xb.numel()
        total_tokens += xb.numel()
    return total_loss / total_tokens

rnn_history = {"train_loss": [], "val_loss": []}
for epoch in range(1, 6):
    train_loss = train_rnn_epoch(rnn_model, train_loader_rnn)
    val_loss = eval_rnn_epoch(rnn_model, val_loader_rnn)
    rnn_history["train_loss"].append(round(train_loss, 4))
    rnn_history["val_loss"].append(round(val_loss, 4))
    print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

@torch.no_grad()
def generate_text(model, seed_text="ROMEO:\n", length=220, temperature=0.8):
    model.eval()
    input_ids = torch.tensor([[stoi[ch] for ch in seed_text]], dtype=torch.long, device=device)
    generated = list(seed_text)
    hidden = None

    for _ in range(length):
        logits, hidden = model(input_ids, hidden)
        next_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        next_char = itos[int(next_id.item())]
        generated.append(next_char)
        input_ids = next_id
    return "".join(generated)

generated_text = generate_text(rnn_model)
print(generated_text)

plt.figure(figsize=(6, 4))
epochs = range(1, len(rnn_history["train_loss"]) + 1)
plt.plot(epochs, rnn_history["train_loss"], label="Train loss")
plt.plot(epochs, rnn_history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RNN/LSTM Loss Over Time")
plt.legend()
plt.tight_layout()
plt.savefig(results_dir / "rnn_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()

(results_dir / "generated_text.txt").write_text(generated_text, encoding="utf-8")

# Save metrics and summary output

metrics = {
    "assumptions": {
        "fashion_subset": "The available Fashion-MNIST CSV files contain 1,000 training rows and 1,000 test rows rather than the full 60,000/10,000 release.",
        "rnn_scope": "The RNN experiment uses a 250,000-character subset of Tiny Shakespeare for faster CPU training."
    },
    "mlp": {
        "test_loss": round(float(mlp_test_loss), 4),
        "test_accuracy": round(float(mlp_test_acc), 4),
        "history": mlp_history,
    },
    "cnn": {
        "test_loss": round(float(cnn_test_loss), 4),
        "test_accuracy": round(float(cnn_test_acc), 4),
        "history": cnn_history,
        "confusion_matrix": cm.tolist(),
    },
    "rnn": {
        "final_train_loss": rnn_history["train_loss"][-1],
        "final_val_loss": rnn_history["val_loss"][-1],
        "val_perplexity": round(float(math.exp(rnn_history["val_loss"][-1])), 4),
        "history": rnn_history,
        "generated_text": generated_text,
    }
}

(results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2))