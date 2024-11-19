import numpy as np
import torch
from torch.utils.data import Dataset


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


class Hartmann3D(Dataset):
    """Hartmann 3D function
    Input dimension: 3
    Domain: x_i ∈ [0, 1] for i = 1, 2, 3
    Global minimum: f(0.114614, 0.555649, 0.852547) ≈ -3.86278
    """

    def __init__(self, n_samples):
        self.X = torch.rand(n_samples, 3)
        alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])
        A = torch.tensor([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
        P = torch.tensor(
            [
                [0.3689, 0.1170, 0.2673],
                [0.4699, 0.4387, 0.7470],
                [0.1091, 0.8732, 0.5547],
                [0.0381, 0.5743, 0.8828],
            ]
        )

        self.y = torch.zeros(n_samples)
        for i in range(n_samples):
            sum_val = 0
            for j in range(4):
                inner_sum = 0
                for k in range(3):
                    inner_sum += A[j, k] * (self.X[i, k] - P[j, k]) ** 2
                sum_val += alpha[j] * torch.exp(-inner_sum)
            self.y[i] = -sum_val

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Ackley5D(Dataset):
    """Ackley function in 5D
    Input dimension: 5
    Domain: x_i ∈ [-32.768, 32.768] for i = 1,...,5
    Global minimum: f(0,...,0) = 0
    """

    def __init__(self, n_samples):
        self.X = torch.rand(n_samples, 5) * 65.536 - 32.768
        a, b, c = 20, 0.2, 2 * np.pi

        self.y = torch.zeros(n_samples)
        for i in range(n_samples):
            term1 = -a * torch.exp(-b * torch.sqrt(torch.mean(self.X[i] ** 2)))
            term2 = -torch.exp(torch.mean(torch.cos(c * self.X[i])))
            self.y[i] = term1 + term2 + a + np.exp(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Michalewicz7D(Dataset):
    """Michalewicz function in 7D
    Input dimension: 7
    Domain: x_i ∈ [0, π] for i = 1,...,7
    Has steep valleys and multiple local minima
    """

    def __init__(self, n_samples):
        self.X = torch.rand(n_samples, 7) * np.pi
        m = 10  # steepness factor

        self.y = torch.zeros(n_samples)
        for i in range(n_samples):
            sum_val = 0
            for j in range(7):
                sum_val += torch.sin(self.X[i, j]) * torch.sin(
                    ((j + 1) * self.X[i, j] ** 2) / np.pi
                ) ** (2 * m)
            self.y[i] = -sum_val

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Levy10D(Dataset):
    """Levy function in 10D
    Input dimension: 10
    Domain: x_i ∈ [-10, 10] for i = 1,...,10
    Global minimum: f(1,...,1) = 0
    """

    def __init__(self, n_samples):
        self.X = torch.rand(n_samples, 10) * 20 - 10

        self.y = torch.zeros(n_samples)
        for i in range(n_samples):
            w = 1 + (self.X[i] - 1) / 4

            term1 = torch.sin(np.pi * w[0]) ** 2

            term2 = torch.sum(
                (w[:-1] - 1) ** 2 * (1 + 10 * torch.sin(np.pi * w[:-1] + 1) ** 2)
            )

            term3 = (w[-1] - 1) ** 2 * (1 + torch.sin(2 * np.pi * w[-1]) ** 2)

            self.y[i] = term1 + term2 + term3

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
