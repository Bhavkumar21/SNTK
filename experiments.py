import torch
import numpy as np
from data import generate_data
from models import SimpleNN
from training import train_with_sgd, train_with_sntk, train_with_gd, train_with_ntk
from visualization import plot_loss_comparison, plot_evolution, plot_four_method_comparison


def run_width_experiment(widths):
    """Run experiment with different network widths using all four methods"""
    results = {}
    
    # Generate data
    X, y = generate_data(n_samples=100)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Generate test points
    X_test = np.linspace(-4, 4, 100).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_true = np.sin(X_test)
    
    for width in widths:
        print(f"\n===== Training models with width = {width} =====")

        # HYPER PARAMETERS
        batch_size = 5
        lr = 0.001 * (128/width)**0.5

        # Initialize models with identical weights
        model_gd = SimpleNN(width=width)
        model_ntk = SimpleNN(width=width)
        model_sgd = SimpleNN(width=width)
        model_sntk = SimpleNN(width=width)
        
        # Save initial state for reuse
        state_dict = model_gd.state_dict()
        model_ntk.load_state_dict(state_dict)
        model_sgd.load_state_dict(state_dict)
        model_sntk.load_state_dict(state_dict)
        
        # First train SGD with incremental epochs until convergence
        epochs, sgd_losses, sgd_preds = train_with_sgd(
            model_sgd, X_tensor, y_tensor, X_test_tensor,
            initial_epochs=100, batch_size=10, lr=lr,
            patience=5, min_delta=1e-4
        )
        
        print(f"Training all other methods for {epochs} epochs")
        
        gd_losses, gd_preds = train_with_gd(
            model_gd, X_tensor, y_tensor, X_test_tensor,
            epochs=epochs, lr=lr        
        )
        
        ntk_losses, ntk_preds = train_with_ntk(
            model_ntk, X_tensor, y_tensor, X_test_tensor,
            epochs=epochs, lr=lr
        )
        
        sntk_losses, sntk_preds = train_with_sntk(
            model_sntk, X_tensor, y_tensor, X_test_tensor,
            epochs=epochs, batch_size=batch_size, lr=lr
        )
        
        num_values = 5
        val = [int(round(i * (epochs-1) / (num_values-1))) for i in range(num_values)]
        # Plot individual width results
        plot_loss_comparison(sgd_losses, sntk_losses, name=f"{width}")
        plot_evolution(X_test, y_test_true, sgd_preds, sntk_preds, val, name=f"{width}")
        plot_four_method_comparison(
            X, y, X_test, y_test_true,
            gd_losses, ntk_losses, sgd_losses, sntk_losses,
            gd_preds, ntk_preds, sgd_preds, sntk_preds,
            width
        )
    
    return results


