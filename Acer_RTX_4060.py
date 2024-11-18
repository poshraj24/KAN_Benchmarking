import numpy as np, torch, torch.nn as nn, wandb
from torch.utils.data import Dataset, DataLoader
from kan import KAN
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import time, psutil, warnings, os

# CUDA settings
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True


class FrankeBenchmarkDataset(Dataset):
    # number of inputs: 2
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
        self.config = config
        self.model = KAN(**config, device=self.device)
        self.model.to(self.device)
        try:
            wandb.finish()
        except:
            pass

    def _get_system_metrics(self):
        metrics = {
            "ram_usage_gb": psutil.Process().memory_info().rss / 1024**3,
            "ram_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(),
        }
        if torch.cuda.is_available():
            metrics.update(
                {
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                    "gpu_max_memory_allocated_gb": torch.cuda.max_memory_allocated()
                    / 1024**3,
                    "gpu_utilization": torch.cuda.utilization(),
                }
            )
        return metrics

    def _get_model_metrics(self):
        return {
            "params_count": sum(p.numel() for p in self.model.parameters()),
            "model_size_mb": sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            )
            / 1024**2,
        }

    def train_and_evaluate(self, n_samples=1000, batch_size=32, lr=0.01, epochs=20):
        experiment_name = (
            f"KAN_Experiment_{self.config['width'][0]}to{self.config['width'][-1]}"
        )
        results = {}

        try:
            wandb.init(
                project="kan-benchmarking",
                name=experiment_name,
                config={**self._get_model_metrics(), **self.config},
                reinit=True,
            )

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
                train_true, train_pred = [], []

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()

                for X, y in train_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    try:
                        output = self.model(X)
                        loss = criterion(output.squeeze(), y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                        train_pred.extend(output.detach().cpu().numpy().squeeze())
                        train_true.extend(y.cpu().numpy())
                    except RuntimeError as e:
                        print(f"Error in training: {e}")
                        continue

                train_true = np.array(train_true)
                train_pred = np.array(train_pred)

                # Log training metrics and system utilization
                metrics = {
                    "epoch": epoch,
                    "train_loss": total_loss / len(train_loader),
                    "train_mse": mean_squared_error(train_true, train_pred),
                    "train_rmse": np.sqrt(mean_squared_error(train_true, train_pred)),
                    "train_mae": mean_absolute_error(train_true, train_pred),
                    "train_r2": r2_score(train_true, train_pred),
                    **self._get_system_metrics(),  # System metrics per epoch
                }
                wandb.log(metrics)

            # Final evaluation
            self.model.eval()
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

                results.update(
                    {
                        f"{split}_mse": mean_squared_error(y_true, y_pred),
                        f"{split}_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                        f"{split}_r2": r2_score(y_true, y_pred),
                        f"{split}_mae": mean_absolute_error(y_true, y_pred),
                        f"{split}_mape": np.mean(np.abs((y_true - y_pred) / y_true))
                        * 100,
                    }
                )

            # Log final metrics including training time and system utilization
            results.update(
                {
                    "training_time": time.time() - start_time,
                    **self._get_model_metrics(),
                    **self._get_system_metrics(),  # Final system metrics
                }
            )

            wandb.log(results)
            wandb.finish()
            return results

        except Exception as e:
            print(f"Error during training: {e}")
            wandb.finish()
            raise e


if __name__ == "__main__":
    config = {"width": [2, 3, 4, 1], "grid": 6, "k": 5, "seed": 42}
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    benchmark = KANBenchmark(config)
    try:
        metrics = benchmark.train_and_evaluate(epochs=20)
        print("\nKAN Benchmark Results")
        print(f"Parameters: {metrics['params_count']:,}")
        print(f"Model Size: {metrics['model_size_mb']:.2f} MB")
        print(f"Training Time: {metrics['training_time']:.2f}s")
        print(f"\nFinal Performance Metrics:")
        print(f"Training MSE: {metrics['train_mse']:.6f}")
        print(f"Training RMSE: {metrics['train_rmse']:.6f}")
        print(f"Training R²: {metrics['train_r2']:.4f}")
        print(f"Test MSE: {metrics['test_mse']:.6f}")
        print(f"Test RMSE: {metrics['test_rmse']:.6f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"\nSystem Utilization:")
        print(f"RAM Usage: {metrics['ram_usage_gb']:.2f}GB ({metrics['ram_percent']}%)")
        print(f"CPU Usage: {metrics['cpu_percent']}%")
        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {metrics['gpu_memory_allocated_gb']:.2f}GB")
            print(f"GPU Memory Reserved: {metrics['gpu_memory_reserved_gb']:.2f}GB")
            print(f"GPU Peak Memory: {metrics['gpu_max_memory_allocated_gb']:.2f}GB")
            print(f"GPU Utilization: {metrics['gpu_utilization']}%")
    except Exception as e:
        print(f"Benchmark failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
