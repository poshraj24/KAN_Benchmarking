import numpy as np, torch, torch.nn as nn, wandb
from torch.utils.data import Dataset, DataLoader
from kan import KAN
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
)
import time, psutil, warnings, os

# Set CUDA settings and suppress warnings
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True


class FrankeBenchmarkDataset(Dataset):
    def __init__(self, n_samples):
        self.X = torch.rand(n_samples, 2)
        self.y = torch.tensor(
            [
                0.75 * np.exp(-((9 * x - 2) ** 2) / 4 - ((9 * y - 2) ** 2) / 4)
                + 0.75 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
                + 0.5 * np.exp(-((9 * x - 7) ** 2) / 4 - ((9 * y - 3) ** 2) / 4)
                - 0.2 * np.exp(-((9 * x - 4) ** 2) - ((9 * y - 7) ** 2))
                for x, y in zip(self.X[:, 0], self.X[:, 1])
            ]
        ).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class KANBenchmark:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model = KAN(**config, device=self.device)
        self.model.to(self.device)
        try:
            wandb.finish()
        except:
            pass

    def _get_metrics(self):
        metrics = {
            "params": sum(p.numel() for p in self.model.parameters()),
            "model_size_mb": sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            )
            / 1024**2,
            "ram_gb": psutil.Process().memory_info().rss / 1024**3,
        }
        if torch.cuda.is_available():
            metrics["gpu_gb"] = torch.cuda.memory_allocated() / 1024**3
        return metrics

    def train_and_evaluate(self, n_samples=1000, batch_size=32, lr=0.01, epochs=20):
        wandb.init(project="kan-benchmarking", config=self._get_metrics(), reinit=True)
        train_loader = DataLoader(
            FrankeBenchmarkDataset(n_samples), batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            FrankeBenchmarkDataset(n_samples // 4), batch_size=batch_size
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        start_time = time.time()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                try:
                    output = self.model(X)
                    loss = criterion(output.squeeze(), y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                except RuntimeError as e:
                    print(f"Error in training: {e}")
                    continue

            avg_loss = total_loss / len(train_loader)
            wandb.log({"epoch": epoch, "loss": avg_loss, **self._get_metrics()})

        self.model.eval()
        results = {}
        for split, loader in [("train", train_loader), ("test", test_loader)]:
            y_true, y_pred = [], []
            with torch.no_grad():
                for X, y in loader:
                    X = X.to(self.device)
                    try:
                        pred = self.model(X).cpu().squeeze()
                        y_true.extend(y.numpy())
                        y_pred.extend(pred.numpy())
                    except RuntimeError as e:
                        print(f"Error in evaluation: {e}")
                        continue

            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            y_true_bin = (y_true > y_true.mean()).astype(int)
            y_pred_bin = (y_pred > y_pred.mean()).astype(int)

            results.update(
                {
                    f"{split}_mse": mean_squared_error(y_true, y_pred),
                    f"{split}_r2": r2_score(y_true, y_pred),
                    f"{split}_accuracy": accuracy_score(y_true_bin, y_pred_bin),
                    f"{split}_precision": precision_score(y_true_bin, y_pred_bin),
                    f"{split}_recall": recall_score(y_true_bin, y_pred_bin),
                }
            )

        results.update(
            {"training_time": time.time() - start_time, **self._get_metrics()}
        )
        wandb.log(results)
        wandb.finish()
        return results


if __name__ == "__main__":
    config = {"width": [2, 4, 2, 1], "grid": 3, "k": 3, "seed": 42}
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    benchmark = KANBenchmark(config)
    try:
        metrics = benchmark.train_and_evaluate(epochs=20)
        print("\nKAN Benchmark Results")
        print(f"Parameters: {metrics['params']:,}")
        print(f"Model Size: {metrics['model_size_mb']:.2f} MB")
        print(f"Training Time: {metrics['training_time']:.2f}s")
        print(f"Final Training MSE: {metrics['train_mse']:.6f}")
        print(f"Final Test MSE: {metrics['test_mse']:.6f}")
        print(
            f"Train/Test Accuracy: {metrics['train_accuracy']:.2%}/{metrics['test_accuracy']:.2%}"
        )
        print(f"Memory Usage - RAM: {metrics['ram_gb']:.2f}GB")
        if "gpu_gb" in metrics:
            print(f"GPU Memory: {metrics['gpu_gb']:.2f}GB")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
