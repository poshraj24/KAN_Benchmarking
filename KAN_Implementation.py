import torch
import torch.nn as nn
import numpy as np
import wandb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from torch.utils.data import DataLoader
from kan import KAN
import time, psutil, os
from datetime import datetime
from Datasets import (
    FrankeBenchmarkDataset,
    Hartmann3D,
    Ackley5D,
    Michalewicz7D,
    Levy10D,
)


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

    def _write_log_to_file(self, metrics, experiment_name, batch_size, lr, epochs):
        """Write experiment results to a log file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "experiment_logs"
        os.makedirs(log_dir, exist_ok=True)

        filename = f"{log_dir}/{experiment_name}_{timestamp}.txt"

        with open(filename, "w") as f:
            f.write(f"KAN Benchmark Results - {experiment_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Configuration: {self.config}\n\n")

            f.write(f"Training Hyperparameters:\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Learning Rate: {lr}\n")
            f.write(f"Epochs: {epochs}\n\n")

            f.write(f"Model Information:\n")
            f.write(f"Parameters: {metrics['params_count']:,}\n")
            f.write(f"Model Size: {metrics['model_size_mb']:.2f} MB\n")
            f.write(f"Training Time: {metrics['training_time']:.2f}s\n\n")

            f.write(f"Performance Metrics:\n")
            f.write(f"Training MSE: {metrics['train_mse']:.6f}\n")
            f.write(f"Training RMSE: {metrics['train_rmse']:.6f}\n")
            f.write(f"Training R²: {metrics['train_r2']:.4f}\n")
            f.write(f"Test MSE: {metrics['test_mse']:.6f}\n")
            f.write(f"Test RMSE: {metrics['test_rmse']:.6f}\n")
            f.write(f"Test R²: {metrics['test_r2']:.4f}\n\n")

            f.write(f"System Utilization:\n")
            f.write(
                f"RAM Usage: {metrics['ram_usage_gb']:.2f}GB ({metrics['ram_percent']}%)\n"
            )
            f.write(f"CPU Usage: {metrics['cpu_percent']}%\n")

            if torch.cuda.is_available():
                f.write(
                    f"GPU Memory Allocated: {metrics['gpu_memory_allocated_gb']:.2f}GB\n"
                )
                f.write(
                    f"GPU Memory Reserved: {metrics['gpu_memory_reserved_gb']:.2f}GB\n"
                )
                f.write(
                    f"GPU Peak Memory: {metrics['gpu_max_memory_allocated_gb']:.2f}GB\n"
                )
                f.write(f"GPU Utilization: {metrics['gpu_utilization']}%\n")
        return filename

    def train_and_evaluate(
        self,
        n_samples=1000,
        batch_size=64,
        lr=0.05,
        epochs=100,
        train_loader=None,
        test_loader=None,
        dataset_class=None,
    ):
        experiment_name = (
            f"KAN_Experiment_{self.config['width'][0]}to{self.config['width'][-1]}"
        )
        results = {}

        try:
            wandb.init(
                project="kan-benchmarking-Acer_RTX_4060",
                name=experiment_name,
                config={**self._get_model_metrics(), **self.config},
                reinit=True,
            )

            # Use provided loaders or create new ones
            if train_loader is None or test_loader is None:
                train_loader = DataLoader(
                    dataset_class(n_samples), batch_size=batch_size, shuffle=True
                )
                test_loader = DataLoader(
                    dataset_class(n_samples // 4), batch_size=batch_size
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
            # Write results to log file
            log_file = self._write_log_to_file(
                results, experiment_name, batch_size, lr, epochs
            )
            print(f"\nExperiment log saved to: {log_file}")
            return results

        except Exception as e:
            print(f"Error during training: {e}")
            wandb.finish()
            raise e
