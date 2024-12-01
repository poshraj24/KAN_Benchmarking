# Description: Main script to run the KAN benchmark suite on all available datasets.
from Datasets import (
    FrankeBenchmarkDataset,
    Hartmann3D,
    Ackley5D,
    Michalewicz7D,
    Levy10D,
)
from KAN_Implementation import *
from torch.utils.data import DataLoader
import torch
import os
import warnings


# CUDA settings
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True


class DatasetRegistry:
    """Registry of available benchmark datasets"""

    def __init__(self):
        self.datasets = {
            "Franke2D": FrankeBenchmarkDataset,
            "Hartmann3D": Hartmann3D,
            "Ackley5D": Ackley5D,
            "Michalewicz7D": Michalewicz7D,
            "Levy10D": Levy10D,
        }

    def get_dataset(self, name):
        return self.datasets.get(name)

    def get_all_datasets(self):
        return self.datasets

    def get_input_dim(self, name):
        sample_dataset = self.datasets[name](1)
        return sample_dataset.X.shape[1]


def run_benchmark_suite(hyperparameters):
    """Run benchmarks on all available datasets"""
    registry = DatasetRegistry()
    all_results = {}

    # Extract hyperparameters
    n_samples = hyperparameters.get("n_samples", 1000)
    batch_size = hyperparameters.get("batch_size", 80)
    lr = hyperparameters.get("lr", 0.5)
    epochs = hyperparameters.get("epochs", 100)
    hidden_layers = hyperparameters.get("hidden_layers", [12, 12])
    grid = hyperparameters.get("grid", 7)
    k = hyperparameters.get("k", 8)
    seed = hyperparameters.get("seed", 42)

    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Run benchmarks for each dataset
    for dataset_name, dataset_class in registry.datasets.items():
        print(f"\n{'='*50}")
        print(f"Running benchmark for {dataset_name}")
        print(f"{'='*50}")

        try:
            # Get input dimension for current dataset
            input_dim = registry.get_input_dim(dataset_name)

            # Configure model architecture
            config = {
                "width": [input_dim] + hidden_layers + [1],
                "grid": grid,
                "k": k,
                "seed": seed,
            }

            # Initialize benchmark
            benchmark = KANBenchmark(config)

            # Create custom dataset loaders
            train_loader = DataLoader(
                dataset_class(n_samples), batch_size=batch_size, shuffle=True
            )
            test_loader = DataLoader(
                dataset_class(n_samples // 4), batch_size=batch_size
            )

            # Run benchmark
            metrics = benchmark.train_and_evaluate(
                n_samples=n_samples,
                batch_size=batch_size,
                lr=lr,
                epochs=epochs,
                train_loader=train_loader,
                test_loader=test_loader,
                dataset_class=dataset_class,
            )

            all_results[dataset_name] = metrics

            print(f"\nResults for {dataset_name}:")
            print(f"Training R²: {metrics['train_r2']:.4f}")
            print(f"Test R²: {metrics['test_r2']:.4f}")
            print(f"Training Time: {metrics['training_time']:.2f}s")

        except Exception as e:
            print(f"Error running benchmark for {dataset_name}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    return all_results


if __name__ == "__main__":
    # Define hyperparameters for the benchmark suite
    hyperparameters = {
        "n_samples": 1000,
        "batch_size": 80,
        "lr": 0.5,
        "epochs": 100,
        "hidden_layers": [12, 12],  # Hidden layer dimensions
        "grid": 7,
        "k": 8,
        "seed": 42,
    }

    try:
        print("Starting KAN Benchmark Suite")
        print(f"Running with hyperparameters: {hyperparameters}")

        results = run_benchmark_suite(hyperparameters)

        # Print summary of all results
        print("\n" + "=" * 50)
        print("Benchmark Suite Summary")
        print("=" * 50)

        for dataset_name, metrics in results.items():
            print(f"\n{dataset_name}:")
            print(f"Parameters: {metrics['params_count']:,}")
            print(f"Model Size: {metrics['model_size_mb']:.2f} MB")
            print(f"Training Time: {metrics['training_time']:.2f}s")
            print(f"Training R²: {metrics['train_r2']:.4f}")
            print(f"Test R²: {metrics['test_r2']:.4f}")
            print(f"Test RMSE: {metrics['test_rmse']:.6f}")

    except Exception as e:
        print(f"Benchmark suite failed: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
