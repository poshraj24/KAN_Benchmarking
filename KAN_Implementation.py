import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import torch.multiprocessing as tmp
from kan import KAN
import os, time
from sklearn.metrics import mean_squared_error, r2_score
from itertools import product
from Datasets import (
    FrankeBenchmarkDataset,
    Hartmann3D,
    Ackley5D,
    Michalewicz7D,
    Levy10D,
)


class KANScalingAnalysis:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Enable benchmark mode for faster runtime
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        # Define dataset dimensions
        self.dimensions = {
            "Franke_2D": {"class": FrankeBenchmarkDataset, "dim": 2},
            "Hartmann_3D": {"class": Hartmann3D, "dim": 3},
            "Ackley_5D": {"class": Ackley5D, "dim": 5},
            "Michalewicz_7D": {"class": Michalewicz7D, "dim": 7},
            "Levy_10D": {"class": Levy10D, "dim": 10},
        }

        self.setup_gpu_optimization()
        # Test configurations
        self.k_values = [2, 3, 4, 5, 6, 7]
        self.grid_sizes = [3, 4, 5, 6, 7]

        # Training parameters
        self.n_samples = 1000
        self.batch_size = 5048  # Increased for better GPU utilization
        self.lr = 0.5
        self.epochs = 50

        # Parallel processing parameters
        self.num_workers = 4  # DataLoader workers per process
        self.max_parallel_processes = 4
        self.batch_size_configs = 15
        self.prefetch_factor = 2
        self.pin_memory = True

    def setup_gpu_optimization(self):
        """Setup GPU optimization flags"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.enabled = True
            torch.cuda.set_device(0)

            # Optimize memory allocation
            torch.cuda.empty_cache()
            torch.cuda.memory.set_per_process_memory_fraction(
                0.9
            )  # Use 90% of GPU memory

    def get_gpu_memory(self):
        """Get detailed GPU memory statistics"""
        if torch.cuda.is_available():
            # Get memory in bytes and convert to GB
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()

            return {
                "allocated": allocated / (1024**3),
                "reserved": reserved / (1024**3),
                "peak": max_allocated / (1024**3),
            }
        return {"allocated": 0, "reserved": 0, "peak": 0}

    def train_and_measure_memory(self, model, train_loader, test_loader):
        """Optimized training function"""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()

        torch.cuda.reset_peak_memory_stats()
        memory_stats = []
        train_losses = []
        peak_memory = 0

        stream = torch.cuda.Stream()

        # Pre-allocate tensors
        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0

            for batch_idx, (X, y) in enumerate(train_loader):
                with torch.cuda.stream(stream):
                    X = X.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                stream.synchronize()

                with torch.cuda.amp.autocast():
                    output = model(X)
                    loss = criterion(output.squeeze(), y)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % 5 == 0:  # Reduced memory tracking frequency
                    mem_stats = self.get_gpu_memory()
                    memory_stats.append(
                        {
                            "epoch": epoch,
                            "batch": batch_idx,
                            "allocated": mem_stats["allocated"],
                            "reserved": mem_stats["reserved"],
                            "peak": mem_stats["peak"],
                        }
                    )
                    peak_memory = max(peak_memory, mem_stats["peak"])

            train_losses.append(epoch_loss / num_batches)

            if (epoch + 1) % 10 == 0:
                current_mem = self.get_gpu_memory()
                print(f"\nEpoch {epoch+1}/{self.epochs}")
                print(f"Train Loss: {train_losses[-1]:.4f}")
                print(f"Peak Memory: {current_mem['peak']:.3f} GB")

        return {
            "peak_memory": peak_memory,
            "memory_stats": memory_stats,
            "train_losses": train_losses,
            "final_loss": train_losses[-1],
        }

    def process_single_configuration(self, config):
        """Process a single configuration with memory tracking"""
        dataset_name, k, grid = config
        dataset_class = self.dimensions[dataset_name]["class"]
        input_dim = self.dimensions[dataset_name]["dim"]

        try:
            print(f"\nProcessing {dataset_name}: K={k}, Grid={grid}")

            # Create datasets
            train_dataset = dataset_class(self.n_samples)
            test_dataset = dataset_class(self.n_samples // 4)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                prefetch_factor=self.prefetch_factor,
                persistent_workers=True,
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )

            # Configure and create model
            model_config = {
                "width": [input_dim] + [1] + [1],  # 2-3-5-7-10/1-1 architecture
                "grid": grid,
                "k": k,
                "device": self.device,
            }

            model = KAN(**model_config)
            model.to(self.device)

            # Count parameters
            params = sum(p.numel() for p in model.parameters())

            # Train and measure memory
            metrics = self.train_and_measure_memory(model, train_loader, test_loader)

            # Cleanup
            del model
            torch.cuda.empty_cache()

            return {
                "dataset_name": dataset_name,
                "input_dim": input_dim,
                "k_value": k,
                "grid_size": grid,
                "n_params": params,
                "peak_memory_gb": metrics["peak_memory"],
                "memory_stats": metrics["memory_stats"],
                "final_loss": metrics["final_loss"],
            }

        except Exception as e:
            print(f"Error processing {dataset_name} (K={k}, Grid={grid}): {str(e)}")
            return None

    def batch_configurations(self):
        """Generate configuration batches for parallel processing"""
        all_configs = [
            (dataset_name, k, grid)
            for dataset_name, k, grid in product(
                self.dimensions.keys(), self.k_values, self.grid_sizes
            )
        ]

        for i in range(0, len(all_configs), self.batch_size_configs):
            yield all_configs[i : i + self.batch_size_configs]

    def run_analysis(self):
        """Optimized parallel analysis"""
        # Enable GPU optimization flags
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.enabled = True
        all_results = []
        all_configs = list(
            product(self.dimensions.keys(), self.k_values, self.grid_sizes)
        )

        # Process in larger batches
        for i in range(0, len(all_configs), self.batch_size_configs):
            batch = all_configs[i : i + self.batch_size_configs]
            print(
                f"\nProcessing batch {i//self.batch_size_configs + 1}/{len(all_configs)//self.batch_size_configs + 1}"
            )

            with ProcessPoolExecutor(
                max_workers=self.max_parallel_processes
            ) as executor:
                futures = [
                    executor.submit(self.process_single_configuration, config)
                    for config in batch
                ]

                for future in futures:
                    try:
                        result = future.result()
                        if result is not None:
                            all_results.append(result)
                    except Exception as e:
                        print(f"Error in processing: {str(e)}")

            # Optional memory cleanup between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return pd.DataFrame([r for r in all_results if r is not None])

    def visualize_scaling(df, save_path="kan_scaling_analysis.png"):
        """Generate high-quality visualization for LaTeX inclusion."""
        if len(df) == 0:
            print("No data to visualize.")
            return

        # Define figure and axis sizes optimized for LaTeX documents
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 10), dpi=1200, constrained_layout=True
        )

        # Sort datasets by dimension
        unique_datasets = sorted(
            df["dataset_name"].unique(),
            key=lambda x: df[df["dataset_name"] == x]["input_dim"].iloc[0],
        )
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_datasets)))

        # X-axis labels
        x_labels = [
            f"K={k}\nGrid={g}"
            for k in sorted(df["k_value"].unique())
            for g in sorted(df["grid_size"].unique())
        ]
        x = np.arange(len(x_labels))

        # Plot Model Parameters
        for i, dataset in enumerate(unique_datasets):
            subset = df[df["dataset_name"] == dataset]
            ax1.plot(
                x,
                subset["n_params"],
                color=colors[i],
                marker="o",
                label=dataset,
                linewidth=1.2,
            )
        ax1.set_yscale("log")
        ax1.set_ylabel("Number of Parameters", fontsize=12)
        ax1.set_title("Model Parameters by Configuration", fontsize=14)
        ax1.legend(fontsize=10, loc="upper left")
        ax1.grid(True, which="both", linestyle="--", alpha=0.6)

        # Plot Peak Memory Usage
        for i, dataset in enumerate(unique_datasets):
            subset = df[df["dataset_name"] == dataset]
            ax2.plot(
                x,
                subset["peak_memory_gb"],
                color=colors[i],
                marker="s",
                label=dataset,
                linewidth=1.2,
            )
        ax2.set_ylabel("Peak Memory Usage (GB)", fontsize=12)
        ax2.set_title("Peak Memory Usage by Configuration", fontsize=14)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(fontsize=10, loc="upper left")

        # Formatting X-axis
        for ax in [ax1, ax2]:
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Save the figure in high resolution
        plt.savefig(save_path, bbox_inches="tight", dpi=600)
        plt.close()
        print(f"High-quality visualization saved to: {save_path}")

        # Save raw data
        df.to_csv("kan_scaling_metrics.csv", index=False)
        print(f"\nVisualization saved to: {save_path}")
        print("Raw metrics saved to: kan_scaling_metrics.csv")
        print("Detailed memory stats saved to: memory_stats_detailed.csv")
