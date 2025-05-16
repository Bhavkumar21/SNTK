import torch
import numpy as np
from experiments import run_width_experiment

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

if __name__ == "__main__":
    # Run experiment with different network widths
    widths = [1536, 2048, 3072, 4096, 6144]
    results = run_width_experiment(widths)