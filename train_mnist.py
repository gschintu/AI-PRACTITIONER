import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, activation="relu"):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.Sigmoid(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return self.layers(x)


def get_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, test_loader


def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    samples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.shape[0]
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            samples += inputs.shape[0]
    return running_loss / samples, correct / samples


def train(model, loader, optimizer, criterion, device, epoch, scheduler=None):
    model.train()
    total_loss = 0.0
    correct = 0
    samples = 0
    for batch_idx, (inputs, targets) in enumerate(loader, start=1):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        if batch_idx == 1:
            grad_norm = torch.sqrt(
                torch.tensor(sum(param.grad.data.norm(2).item() ** 2 for param in model.parameters() if param.grad is not None))
            )
            print(f"First batch gradient norm: {grad_norm:.4f}")

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * inputs.shape[0]
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        samples += inputs.shape[0]

    return total_loss / samples, correct / samples


def save_model(model, save_dir, run_name):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"mnist_model_{run_name}.pt")
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Train a simple MNIST model and review DL concepts.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--activation", choices=["relu", "sigmoid"], default="relu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--save-dir", type=str, default="./runs")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_data(batch_size=args.batch_size)
    model = SimpleMLP(hidden_dim=args.hidden_dim, activation=args.activation).to(device)
    model.apply(init_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print("Training configuration:")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, momentum={args.momentum}")
    print(f"  activation={args.activation}, loss=CrossEntropyLoss, optimizer=SGD")
    print("  scheduler=StepLR(step_size=5, gamma=0.5)")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, epoch, scheduler)
        valid_loss, valid_acc = evaluate(model, test_loader, device, criterion)
        epoch_time = time.time() - start_time

        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={valid_loss:.4f} val_acc={valid_acc:.4f}  "
              f"time={epoch_time:.2f}s")

        if epoch > 1 and valid_loss > train_loss * 1.5:
            print("Warning: validation loss is much higher than training loss. Check for overfitting or data issues.")

    run_name = f"{int(time.time())}"
    save_model(model, args.save_dir, run_name)
    print("Training complete.")


if __name__ == "__main__":
    main()
