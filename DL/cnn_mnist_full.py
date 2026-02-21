# cnn_mnist_full.py
# Run: python3 cnn_mnist_full.py
# Installs (if needed): pip install torch torchvision matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / max(1, len(loader))
    acc = correct / max(1, total)
    return avg_loss, acc


def show_predictions(model, loader, device, n=8):
    model.eval()
    x, y = next(iter(loader))
    x, y = x[:n].to(device), y[:n].to(device)

    with torch.no_grad():
        preds = model(x).argmax(dim=1)

    x_cpu = x.cpu().squeeze(1).numpy()
    y_cpu = y.cpu().numpy()
    p_cpu = preds.cpu().numpy()

    plt.figure(figsize=(10, 4))
    for i in range(n):
        plt.subplot(2, 4, i + 1)
        plt.imshow(x_cpu[i], cmap="gray")
        plt.title(f"pred={p_cpu[i]}, true={y_cpu[i]}")
        plt.axis("off")
    plt.suptitle("MNIST Predictions")
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 80)
    print("CNN ON MNIST - FULL TRAINING SCRIPT")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Data
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    # Model
    model = SimpleCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 3  # change to 5 or 10 for better accuracy

    # Before training
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Before training -> Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Train
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        test_loss, test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch+1}/{epochs} -> "
              f"Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    # Show a few predictions
    show_predictions(model, test_loader, device, n=8)

    # Optional: Save model
    torch.save(model.state_dict(), "cnn_mnist.pth")
    print("Saved model to cnn_mnist.pth")

    print("Done.")


if __name__ == "__main__":
    main()