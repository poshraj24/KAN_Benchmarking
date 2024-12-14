import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from kan import KAN
import os, time
from sklearn.metrics import mean_squared_error, r2_score
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

        # Define dataset dimensions
        self.dimensions = {
            "Franke_2D": {"class": FrankeBenchmarkDataset, "dim": 2},
            "Hartmann_3D": {"class": Hartmann3D, "dim": 3},
            "Ackley_5D": {"class": Ackley5D, "dim": 5},
            "Michalewicz_7D": {"class": Michalewicz7D, "dim": 7},
            "Levy_10D": {"class": Levy10D, "dim": 10},
        }

        # Test configurations
        self.k_values = [2, 3, 4, 5, 6, 7]
        self.grid_sizes = [3, 4, 5, 6, 7]

        # Training parameters
        self.n_samples = 1000
        self.batch_size = 256
        self.lr = 0.5
        self.epochs = 50

    def train_and_measure_memory(self, model, train_loader, test_loader):
        """Train model and track memory usage during training"""
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Memory tracking
        peak_memory = 0
        memory_usage = []
        train_losses = []

        print("\nTraining and measuring memory...")

        for epoch in range(self.epochs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            # Training phase
            model.train()
            epoch_loss = 0
            num_batches = 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = model(X)
                loss = criterion(output.squeeze(), y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated() / 1024**3
                    peak_memory = max(
                        peak_memory, torch.cuda.max_memory_allocated() / 1024**3
                    )
                    memory_usage.append(current_memory)

            # Calculate the loss for the epoch
            train_losses.append(epoch_loss / num_batches)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}")
                print(f"Train Loss: {train_losses[-1]:.4f}")
                print(f"Current Memory: {current_memory:.2f} GB")
                print(f"Peak Memory: {peak_memory:.2f} GB")
                print("-" * 30)

        return {
            "peak_memory": peak_memory,
            "memory_timeline": memory_usage,
            "train_losses": train_losses,
            "final_loss": train_losses[-1],
        }

    def run_analysis(self):
        all_results = []

        for dataset_name, dataset_info in self.dimensions.items():
            print(f"\nAnalyzing {dataset_name}")
            dataset_class = dataset_info["class"]
            input_dim = dataset_info["dim"]

            try:
                # Create datasets
                train_dataset = dataset_class(self.n_samples)
                test_dataset = dataset_class(self.n_samples // 4)
                train_loader = DataLoader(
                    train_dataset, batch_size=self.batch_size, shuffle=True
                )
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

                for k in self.k_values:
                    for grid in self.grid_sizes:
                        print(f"\nConfiguration: K={k}, Grid={grid}")

                        try:
                            # Configure and initialize model
                            config = {
                                "width": [input_dim] + [k] * 3 + [1],
                                "grid": grid,
                                "k": k,
                                "device": self.device,
                            }
                            model = KAN(**config)
                            model.to(self.device)

                            # Count parameters
                            params = sum(p.numel() for p in model.parameters())

                            # Train and measure memory
                            memory_metrics = self.train_and_measure_memory(
                                model, train_loader, test_loader
                            )

                            # Store results
                            result = {
                                "dataset_name": dataset_name,
                                "input_dim": input_dim,
                                "k_value": k,
                                "grid_size": grid,
                                "n_params": params,
                                "peak_memory_gb": memory_metrics["peak_memory"],
                                "memory_timeline": memory_metrics["memory_timeline"],
                                "final_loss": memory_metrics["final_loss"],
                            }
                            all_results.append(result)

                            print("\nResults:")
                            print(f"Parameters: {params:,}")
                            print(
                                f"Peak Memory: {memory_metrics['peak_memory']:.2f} GB"
                            )
                            print(f"Final Loss: {memory_metrics['final_loss']:.4f}")
                            print("-" * 50)

                            # Cleanup
                            del model
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        except Exception as e:
                            print(f"Error: {str(e)}")
                            continue

            except Exception as e:
                print(f"Error with {dataset_name}: {str(e)}")
                continue

        return pd.DataFrame(all_results)

    def visualize_scaling(self, df, save_path="kan_scaling_analysis.png"):
        if len(df) == 0:
            print("No data to visualize.")
            return

        plt.figure(figsize=(15, 8))

        # Main axes for parameters
        ax1 = plt.gca()
        # Twin axes for memory
        ax2 = ax1.twinx()

        # Sort datasets by dimension
        unique_datasets = sorted(
            df["dataset_name"].unique(), key=lambda x: self.dimensions[x]["dim"]
        )
        colors = plt.cm.viridis(np.linspace(0, 1, len(unique_datasets)))

        # X-axis configuration
        x = np.arange(len(self.k_values) * len(self.grid_sizes))
        x_labels = [f"K={k}\nGrid={g}" for k in self.k_values for g in self.grid_sizes]

        # Plot data
        for i, dataset in enumerate(unique_datasets):
            subset = df[df["dataset_name"] == dataset]

            # Parameters (lines with markers)
            ax1.plot(
                x,
                subset["n_params"],
                color=colors[i],
                linestyle="-",
                marker="o",
                label=f"{dataset} (Params)",
            )

            # Memory (dashed lines with different markers)
            ax2.plot(
                x,
                subset["peak_memory_gb"],
                color=colors[i],
                linestyle="--",
                marker="s",
                label=f"{dataset} (Memory)",
            )

        # Customize axes
        ax1.set_yscale("log")
        ax2.set_yscale("log")

        ax1.set_ylabel("Number of Parameters (log scale)", fontsize=12)
        ax2.set_ylabel("Peak GPU Memory (GB, log scale)", fontsize=12)

        plt.xticks(x, x_labels, rotation=45)
        plt.xlabel("Model Configuration (K values and Grid sizes)", fontsize=12)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left",
            bbox_to_anchor=(1.05, 1),
        )

        plt.title("KAN Scaling Analysis: Parameters and Memory Usage", fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()

        # Save raw data
        df.to_csv("kan_scaling_metrics.csv", index=False)
        print(f"\nVisualization saved to: {save_path}")
        print("Raw metrics saved to: kan_scaling_metrics.csv")
