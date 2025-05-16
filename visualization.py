import matplotlib.pyplot as plt
import numpy as np
from kernels import calculate_rmse

def plot_loss_comparison(sgd_losses, sntk_losses, name):
    """Plot loss curves for SGD and SNTK methods"""
    plt.figure(figsize=(10, 6))
    plt.plot(sgd_losses, color='blue', label='SGD Loss')
    plt.plot(sntk_losses, color='red', label='SNTK Loss')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss Comparison: SGD vs SNTK', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'img/loss_comparison-{name}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_evolution(X_test, y_test_true, sgd_preds, sntk_preds, epochs_to_plot, name):
    """Plot the evolution of predictions over training"""
    sgd_cmap = plt.cm.Blues
    sntk_cmap = plt.cm.Reds
    num_epochs = len(epochs_to_plot)
    sgd_colors = [sgd_cmap(0.3 + 0.7 * i/num_epochs) for i in range(num_epochs)]
    sntk_colors = [sntk_cmap(0.3 + 0.7 * i/num_epochs) for i in range(num_epochs)]
    
    plt.figure(figsize=(12, 8))
    
    # Plot SGD evolution
    plt.subplot(1, 2, 1)
    for i, epoch in enumerate(epochs_to_plot):
        plt.plot(X_test, sgd_preds[epoch], color=sgd_colors[i], 
                label=f'Epoch {epoch}')
    plt.plot(X_test, y_test_true, 'k--', linewidth=2, label='True Function')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('SGD Prediction Evolution', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Plot SNTK evolution
    plt.subplot(1, 2, 2)
    for i, epoch in enumerate(epochs_to_plot):
        plt.plot(X_test, sntk_preds[epoch], color=sntk_colors[i],
                label=f'Epoch {epoch}')
    plt.plot(X_test, y_test_true, 'k--', linewidth=2, label='True Function')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('SNTK Prediction Evolution', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'img/evolution_comparison-{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_four_method_comparison(X, y, X_test, y_test_true, 
                              gd_losses, ntk_losses, sgd_losses, sntk_losses,
                              gd_preds, ntk_preds, sgd_preds, sntk_preds,
                              width):
    """Create comprehensive visualizations comparing all four methods"""
    # Create a figure with a 2x2 grid of subplots
    plt.figure(figsize=(15, 12))
    
    # 1. LOSS COMPARISON - Top left
    plt.subplot(2, 2, 1)
    plt.plot(gd_losses, color='blue', linewidth=1.5, label='GD')
    plt.plot(ntk_losses, color='green', linewidth=1.5, label='NTK') 
    plt.plot(sgd_losses, color='red', linewidth=1.5, label='SGD')
    plt.plot(sntk_losses, color='purple', linewidth=1.5, label='SNTK')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.yscale('log')  # Log scale often better shows convergence differences
    plt.title('Loss Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 2. PREDICTION COMPARISON - Top right
    plt.subplot(2, 2, 2)
    # Add shaded regions to highlight training vs extrapolation
    plt.axvspan(-3, 3, color='lightyellow', alpha=0.3, label='Training Region')
    plt.axvspan(-4, -3, color='lightgray', alpha=0.3, label='Extrapolation')
    plt.axvspan(3, 4, color='lightgray', alpha=0.3)
    
    plt.scatter(X, y, c='black', s=20, alpha=0.6, label='Training Data')
    plt.plot(X_test, y_test_true, color='black', linestyle='--', linewidth=2, label='True Function')
    plt.plot(X_test, gd_preds[-1], color='blue', linewidth=1.5, label='GD')
    plt.plot(X_test, ntk_preds[-1], color='green', linewidth=1.5, label='NTK')
    plt.plot(X_test, sgd_preds[-1], color='red', linewidth=1.5, label='SGD')
    plt.plot(X_test, sntk_preds[-1], color='purple', linewidth=1.5, label='SNTK')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Final Predictions', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 3. TRAINING DYNAMICS - Bottom left (deterministic methods)
    plt.subplot(2, 2, 3)
    epochs = 100
    epochs_to_show = [0, int(epochs*0.25), int(epochs*0.5), int(epochs*0.75), epochs-1]
    colors_gd = plt.cm.Blues(np.linspace(0.3, 1.0, len(epochs_to_show)))
    colors_ntk = plt.cm.Greens(np.linspace(0.3, 1.0, len(epochs_to_show)))
    
    for i, epoch in enumerate(epochs_to_show):
        plt.plot(X_test, gd_preds[epoch], color=colors_gd[i], 
                 linewidth=1.2, label=f'GD Epoch {epoch}')
        plt.plot(X_test, ntk_preds[epoch], color=colors_ntk[i], 
                 linewidth=1.2, linestyle='--', label=f'NTK Epoch {epoch}')
    
    plt.plot(X_test, y_test_true, 'k--', linewidth=1.5)
    plt.scatter(X, y, c='black', s=15, alpha=0.5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('GD and NTK Training Evolution', fontsize=14)
    plt.legend(fontsize=9, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 4. TRAINING DYNAMICS - Bottom right (stochastic methods)
    plt.subplot(2, 2, 4)
    colors_sgd = plt.cm.Reds(np.linspace(0.3, 1.0, len(epochs_to_show)))
    colors_sntk = plt.cm.Purples(np.linspace(0.3, 1.0, len(epochs_to_show)))
    
    for i, epoch in enumerate(epochs_to_show):
        plt.plot(X_test, sgd_preds[epoch], color=colors_sgd[i], 
                 linewidth=1.2, label=f'SGD Epoch {epoch}')
        plt.plot(X_test, sntk_preds[epoch], color=colors_sntk[i], 
                 linewidth=1.2, linestyle='--', label=f'SNTK Epoch {epoch}')
    
    plt.plot(X_test, y_test_true, 'k--', linewidth=1.5)
    plt.scatter(X, y, c='black', s=15, alpha=0.5)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('SGD and SNTK Training Evolution', fontsize=14) 
    plt.legend(fontsize=9, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle(f'Comparison of Training Methods (Width = {width})', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(f'img/four_method_comparison-{width}-5bt.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Focused view: only final predictions
    plt.figure(figsize=(10, 6))
    plt.axvspan(-3, 3, color='lightyellow', alpha=0.3, label='Training Region')
    plt.axvspan(-4, -3, color='lightgray', alpha=0.3, label='Extrapolation')
    plt.axvspan(3, 4, color='lightgray', alpha=0.3)
    plt.plot(X_test, y_test_true, color='black', linestyle='--', linewidth=2, label='True Function')
    plt.plot(X_test, gd_preds[-1], color='blue', linewidth=2, label='GD')
    plt.plot(X_test, ntk_preds[-1], color='green', linewidth=2, label='NTK')
    plt.plot(X_test, sgd_preds[-1], color='red', linewidth=2, label='SGD')
    plt.plot(X_test, sntk_preds[-1], color='purple', linewidth=2, label='SNTK')
    plt.scatter(X, y, c='black', s=30, alpha=0.6, label='Training Data')
    
    # Calculate RMSE between methods for comparison
    sgd_vs_sntk = calculate_rmse(sgd_preds[-1], sntk_preds[-1])
    sgd_vs_ntk = calculate_rmse(sgd_preds[-1], ntk_preds[-1])
    sgd_vs_gd = calculate_rmse(sgd_preds[-1], gd_preds[-1])
    
    # Create metrics text for bottom right corner
    metrics_text = (
        f"Method Similarity:\n"
        f"SGD vs SNTK: {sgd_vs_sntk:.4f}\n"
        f"SGD vs NTK: {sgd_vs_ntk:.4f}\n"
        f"SGD vs GD: {sgd_vs_gd:.4f}"
    )
    
    # Add metrics text to the bottom right corner
    plt.text(0.98, 0.02, metrics_text, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(f'Prediction Comparison (Width={width})', fontsize=16)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'img/four_method_predictions-{width}-5bt.png', dpi=300, bbox_inches='tight')
    plt.close()