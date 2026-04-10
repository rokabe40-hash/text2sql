import argparse
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class AlexNetCIFAR(nn.Module):
    def __init__(self, hidden_units: int = 512, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_accuracy(net: nn.Module, data_iter: DataLoader, device: torch.device) -> float:
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            pred = net(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total if total > 0 else 0.0


def load_data(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)

    total_size = len(dataset)
    test_size = int(total_size * val_ratio)
    train_size = total_size - test_size

    if total_size == 0 or train_size <= 0 or test_size <= 0:
        raise ValueError("数据量不足，请检查 train 目录内容，并确保可按 80/20 划分。")

    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [train_size, test_size], generator=generator)

    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter, dataset.classes


def train(
    net: nn.Module,
    train_iter: DataLoader,
    test_iter: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, List[float]]:
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "epoch_time_sec": [],
    }

    for epoch in range(num_epochs):
        start = time.time()
        net.train()
        metric_loss = 0.0
        metric_correct = 0
        metric_total = 0

        for x, y in train_iter:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = net(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            metric_loss += loss.item() * x.shape[0]
            metric_correct += (y_hat.argmax(dim=1) == y).sum().item()
            metric_total += y.numel()

        train_loss = metric_loss / metric_total
        train_acc = metric_correct / metric_total
        test_acc = evaluate_accuracy(net, test_iter, device)
        elapsed = time.time() - start

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["epoch_time_sec"].append(elapsed)

        print(
            f"epoch {epoch + 1:02d}, "
            f"loss {train_loss:.4f}, "
            f"train acc {train_acc:.4f}, "
            f"test acc {test_acc:.4f}, "
            f"time {elapsed:.2f}s"
        )

    return history


def plot_results(results: Dict[int, Dict[str, List[float]]], output_path: Path) -> None:
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    for hidden_units, history in results.items():
        plt.plot(history["train_loss"], label=f"h={hidden_units}")
    plt.title("Train Loss")
    plt.xlabel("Epoch")

    plt.subplot(1, 3, 2)
    for hidden_units, history in results.items():
        plt.plot(history["train_acc"], label=f"h={hidden_units}")
    plt.title("Train Accuracy")
    plt.xlabel("Epoch")

    plt.subplot(1, 3, 3)
    for hidden_units, history in results.items():
        plt.plot(history["test_acc"], label=f"h={hidden_units}")
    plt.title("Test Accuracy")
    plt.xlabel("Epoch")

    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"曲线已保存到: {output_path}")


def summarize_results(results: Dict[int, Dict[str, List[float]]]) -> Dict[int, Dict[str, float]]:
    summary = {}
    print("\n=== 实验结果汇总 ===")
    print("hidden_units | train_acc | test_acc | acc_gap(train-test) | avg_epoch_time(s)")
    for hidden_units, history in results.items():
        final_train_acc = history["train_acc"][-1]
        final_test_acc = history["test_acc"][-1]
        gap = final_train_acc - final_test_acc
        avg_time = sum(history["epoch_time_sec"]) / len(history["epoch_time_sec"])
        summary[hidden_units] = {
            "final_train_acc": final_train_acc,
            "final_test_acc": final_test_acc,
            "train_test_gap": gap,
            "avg_epoch_time_sec": avg_time,
        }
        print(
            f"{hidden_units:12d} | {final_train_acc:9.4f} | {final_test_acc:8.4f} | "
            f"{gap:19.4f} | {avg_time:15.2f}"
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlexNet on local CIFAR-style train folder (80/20 split)")
    parser.add_argument("--data-dir", type=str, default="train", help="本地数据目录，按 ImageFolder 组织")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--hidden-list", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"未找到数据目录: {data_dir.resolve()}")

    train_iter, test_iter, classes = load_data(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"类别数: {len(classes)}, 类别名: {classes}")
    print(f"训练批次数: {len(train_iter)}, 测试批次数: {len(test_iter)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    results: Dict[int, Dict[str, List[float]]] = {}
    for hidden_units in args.hidden_list:
        print(f"\nTraining hidden_units={hidden_units}")
        net = AlexNetCIFAR(hidden_units=hidden_units, num_classes=len(classes))
        results[hidden_units] = train(
            net=net,
            train_iter=train_iter,
            test_iter=test_iter,
            num_epochs=args.epochs,
            lr=args.lr,
            device=device,
        )

    summary = summarize_results(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_results(results, output_dir / "alexnet_hidden_width_curves.png")

    with open(output_dir / "alexnet_hidden_width_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": vars(args),
                "classes": classes,
                "results": results,
                "summary": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"结果已保存到: {(output_dir / 'alexnet_hidden_width_results.json').resolve()}")


if __name__ == "__main__":
    main()
