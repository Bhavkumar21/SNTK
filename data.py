import numpy as np

def generate_data(n_samples=50):
    """Generate synthetic sine data with noise"""
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    y = np.sin(X) + 0.1 * np.random.randn(n_samples, 1)
    return X, y
