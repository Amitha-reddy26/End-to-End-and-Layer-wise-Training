import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pandas as pd
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.head = nn.Linear(32 * 4 * 4, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.head(self.pool(x).flatten(1))


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    return 100 * correct / len(loader.dataset)


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("Device:", device, flush=True)
    Path("outputs").mkdir(exist_ok=True)
    results = []

    for name, dataset, channels in [
        ("MNIST", datasets.MNIST, 1),
        ("CIFAR10", datasets.CIFAR10, 3)
    ]:
        train = dataset(
            "data", train=True, download=False,
            transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        )
        test = dataset(
            "data", train=False, download=False,
            transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        )

        generator = torch.Generator().manual_seed(42)
        indices = torch.randperm(
            len(train), generator=generator
        )[:5000].tolist()

        train = Subset(train, indices)

        test_loader = DataLoader(
            test, batch_size=256
        )

        for method in ["End-to-End", "Layer-wise"]:
            torch.manual_seed(42)
            train_loader = DataLoader(
                train, batch_size=128, shuffle=True,
                generator=torch.Generator().manual_seed(42)
            )
            model = CNN(channels).to(device)
            start = time.perf_counter()

            for epoch in range(4):
                for p in model.parameters():
                    p.requires_grad = True

                if method == "Layer-wise":
                    if epoch < 2:
                        for p in model.block2.parameters():
                            p.requires_grad = False
                    else:
                        for p in model.block1.parameters():
                            p.requires_grad = False

                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad,
                           model.parameters()),
                    lr=0.001
                )

                loss = train_epoch(
                    model, train_loader, optimizer, device
                )
                print(
                    f"{name} | {method} | Epoch {epoch+1}/4 "
                    f"| Loss: {loss:.4f}", flush=True
                )

            if device.type == "mps":
                torch.mps.synchronize()

            elapsed = time.perf_counter() - start
            accuracy = evaluate(model, test_loader, device)

            results.append({
                "Dataset": name,
                "Method": method,
                "Accuracy": accuracy,
                "Training Loss": loss,
                "Training Time (s)": elapsed
            })

    df = pd.DataFrame(results)
    df.to_csv("outputs/metrics.csv", index=False)
    print(df.to_string(index=False))

    for metric, filename in [
        ("Accuracy", "accuracy.png"),
        ("Training Loss", "loss.png"),
        ("Training Time (s)", "training_time.png")
    ]:
        ax = df.pivot(
            index="Dataset", columns="Method",
            values=metric
        ).plot(kind="bar", rot=0)
        ax.set_ylabel(metric)
        ax.set_title(f"Training Comparison: {metric}")
        plt.tight_layout()
        plt.savefig("outputs/" + filename, dpi=150)
        plt.close()

    print("Training completed. Results saved in outputs/.")


if __name__ == "__main__":
    main()
