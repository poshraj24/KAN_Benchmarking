# KAN-Benchmark: Scaling Analysis for Kolmogorov-Arnold Networks

This repository performs benchmarking and scaling analysis of Kolmogorov-Arnold Networks (KANs) for various test functions of varying dimensionality

- Analyzes KAN performance across 5 standard benchmark functions (2D to 10D)
- Tests multiple spline configurations (order k and grid size)
- Measures parameter count and memory usage
- Provides detailed visualizations of scaling behavior
- Optimized for GPU acceleration

## Installation

1. Clone the repository:
```bash
git clone https://github.com/poshraj24/KAN_Benchmarking.git
cd kan-benchmark
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Dependencies:
```
numpy==1.24.4
torch==2.2.2
matplotlib==3.6.2
pandas==2.0.1
scikit_learn==1.1.3
pykan
pyyaml
tqdm==4.66.2
setuptools==65.5.0
sympy==1.11.1
seaborn
```

## Usage

Run the benchmark with default configurations:
```bash
python main.py
```

The script will:
1. Test KANs across 5 benchmark functions
2. Train models with varying k-order and grid sizes
3. Measure parameter counts and memory usage
4. Generate visualizations and metrics files

## Project Structure

```
KAN-Benchmark/
├── Datasets.py              # Benchmark function implementations
├── KAN_Implementation.py    # KAN scaling analysis implementation
├── main.py                  # Main execution script
├── requirements.txt         # Project dependencies
├── kan_scaling_metrics.csv  # (Generated) Results data
├── kan_scaling_analysis.png # (Generated) Visualization
└── README.md                # Project documentation
```

## Benchmark Functions

The project tests KANs on the following benchmark functions:

- **Franke 2D**: Classic 2D test function with multiple hills and valleys
- **Hartmann 3D**: 3D function with several local minima
- **Ackley 5D**: 5D function with many local minima and a global minimum at the origin
- **Michalewicz 7D**: 7D function with steep valleys and multiple local minima
- **Levy 10D**: 10D function with many local minima

## KAN Configurations

The benchmark tests combinations of:
- **k-order** (2-7): Controls the order of B-splines used in the KAN
- **Grid size** (3-7): Controls the resolution of the spline grid

## Output

The benchmark generates:

1. **kan_scaling_metrics.csv**: Contains detailed metrics for each configuration
2. **kan_scaling_analysis.png**: Visualization showing:
   - Parameter count vs. configuration
   - Peak memory usage vs. configuration

## Advanced Configuration

Modify parameters in `KAN_Implementation.py`:

```python
# Test configurations
self.k_values = [2, 3, 4, 5, 6, 7]
self.grid_sizes = [3, 4, 5, 6, 7]

# Training parameters
self.n_samples = 1000
self.batch_size = 5048
self.lr = 0.5
self.epochs = 50
```

## GPU Acceleration

The code automatically detects and uses GPU acceleration if available. CUDA optimizations are enabled by default.

## Citation

If you use this code in your research, please cite:

```
@article{liu2024kan,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Liu, Ziming and Wang, Yixuan and Vaidya, Sachin and Ruehle, Fabian and Halverson, James and Soljačić, Marin and Hou, Thomas Y. and Tegmark, Max},
  journal={arXiv preprint arXiv:2404.19756v4},
  year={2024}
}
```
