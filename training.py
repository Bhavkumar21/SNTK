import torch
import torch.nn as nn
from tqdm import tqdm
from kernels import compute_ntk, compute_sntk

def train_with_sgd(model, X_tensor, y_tensor, X_test_tensor, initial_epochs=100, batch_size=10, lr=0.01, patience=5, min_delta=1e-5, max_epochs=2000):
    """Train model using SGD with incremental epochs until convergence"""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    predictions = []
    
    # Initial prediction
    with torch.no_grad():
        predictions.append(model(X_test_tensor).numpy())
    
    # Train for initial epochs
    epochs_trained = 0
    converged = False
    
    while not converged and epochs_trained < max_epochs:
        # Determine how many more epochs to train
        epochs_to_add = min(5, max_epochs - epochs_trained)
        
        # Train for additional epochs
        for _ in tqdm(range(epochs_to_add), 
                      desc=f"SGD Training (epochs {epochs_trained+1}-{epochs_trained+epochs_to_add})"):
            # Shuffle data
            indices = torch.randperm(len(X_tensor))
            X_shuffled = X_tensor[indices]
            y_shuffled = y_tensor[indices]
            
            epoch_loss = 0.0
            for i in range(0, len(X_tensor), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # SGD step
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * X_batch.shape[0]
            
            losses.append(epoch_loss / len(X_tensor))
            
            # Record test prediction
            with torch.no_grad():
                predictions.append(model(X_test_tensor).numpy())
        
        # Update total epochs trained
        epochs_trained += epochs_to_add
        
        # Check convergence after we have enough epochs
        if epochs_trained >= patience:
            # Check if loss has stabilized over the last 'patience' epochs
            recent_losses = losses[-patience:]
            min_recent_loss = min(recent_losses)
            max_recent_loss = max(recent_losses)
            
            # Convergence criteria: range of recent losses is small
            loss_range = max_recent_loss - min_recent_loss
            avg_loss = sum(recent_losses) / len(recent_losses)
            relative_change = loss_range / (avg_loss + 1e-10)
            
            if relative_change < min_delta:
                print(f"SGD converged after {epochs_trained} epochs (loss stable at {avg_loss:.6f})")
                converged = True
        
        # Also check for very small loss as convergence indicator
        if losses[-1] < 1e-6:
            print(f"SGD converged after {epochs_trained} epochs (loss below threshold: {losses[-1]:.8f})")
            converged = True
    
    if not converged:
        print(f"SGD reached maximum epochs ({max_epochs}) without convergence")
    
    return epochs_trained, losses, predictions


def train_with_sntk(model, X_tensor, y_tensor, X_test_tensor, epochs=100, batch_size=10, lr=0.01):
    """Train model using SNTK dynamics with improved numerical stability"""
    loss_fn = nn.MSELoss()
    losses = []
    predictions = []
    
    # Get device
    device = next(model.parameters()).device
    
    # Determine width with better default
    width = getattr(model, 'width', 128)
    
    # Initial prediction
    with torch.no_grad():
        predictions.append(model(X_test_tensor).cpu().numpy())
    
    
    for epoch in tqdm(range(epochs), desc="SNTK Training"):
        # Shuffle data
        indices = torch.randperm(len(X_tensor))
        X_shuffled = X_tensor[indices]
        y_shuffled = y_tensor[indices]
        
        # Current predictions and loss
        with torch.no_grad():
            current_outputs = model(X_tensor)
            current_loss = loss_fn(current_outputs, y_tensor).item()
            losses.append(current_loss)
        
        # Adaptive batch size for wider networks
        adaptive_batch_size = min(batch_size, max(5, 100000 // width))
        
        for i in range(0, len(X_tensor), adaptive_batch_size):
            X_batch = X_shuffled[i:i+adaptive_batch_size]
            y_batch = y_shuffled[i:i+adaptive_batch_size]
            
            with torch.no_grad():
                outputs_batch = model(X_batch)
                residuals = outputs_batch - y_batch
            
            for param in model.parameters():
                param.requires_grad_(True)
            
            # SNTK computation and update
            ntk = compute_ntk(model, X_tensor)
            sntk = compute_sntk(model, X_tensor, X_batch, residuals)
            
            with torch.no_grad():
                deterministic_update = -lr * torch.matmul(ntk, (current_outputs - y_tensor))
                noise_scale = torch.sqrt(torch.tensor(lr, device=device))
                
                # Use eigendecomposition instead of Cholesky
                jitter = 1e-4 * (1 + width/512)  # Adaptive regularization
                
                try:
                    # Ensure symmetry
                    sntk_regularized = 0.5 * (sntk + sntk.t()) + jitter * torch.eye(sntk.shape[0], device=device)
                    
                    # More stable eigendecomposition
                    eigenvalues, eigenvectors = torch.linalg.eigh(sntk_regularized)
                    
                    # Ensure positive eigenvalues
                    sqrt_eigenvalues = torch.sqrt(torch.clamp(eigenvalues, min=1e-10))
                    
                    # Generate noise using eigendecomposition
                    stochastic_term = noise_scale * torch.matmul(
                        eigenvectors,
                        torch.matmul(torch.diag(sqrt_eigenvalues),
                                     torch.randn(sntk.shape[0], 1, device=device))
                    )
                except RuntimeError:
                    # Fallback to diagonal approximation
                    print(f"Warning: Using diagonal approximation in epoch {epoch}")
                    sntk_diag = torch.diag(sntk)
                    diagonal_std = torch.sqrt(torch.clamp(sntk_diag, min=1e-10))
                    stochastic_term = noise_scale * diagonal_std.reshape(-1, 1) * torch.randn(sntk.shape[0], 1, device=device)
                
                # Target function values
                target_outputs = current_outputs + deterministic_update + stochastic_term
            
            # Update parameters with adaptive fitting steps
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            # Adaptive fitting steps based on width and loss
            fitting_steps = min(10, max(3, int(5 * min(1.0, current_loss * 10))))
            
            for _ in range(fitting_steps):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                target_fitting_loss = loss_fn(outputs, target_outputs.detach())
                target_fitting_loss.backward()
                optimizer.step()
        
        # Record test prediction
        with torch.no_grad():
            predictions.append(model(X_test_tensor).cpu().numpy())
    
    return losses, predictions

def train_with_gd(model, X_tensor, y_tensor, X_test_tensor, epochs=100, lr=0.01):
    """Train model using standard Gradient Descent (full batch)"""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []
    predictions = []
    
    # Initial prediction
    with torch.no_grad():
        predictions.append(model(X_test_tensor).numpy())
    
    for epoch in tqdm(range(epochs), desc="GD Training"):
        # GD step (full batch)
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = loss_fn(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Record test prediction
        with torch.no_grad():
            predictions.append(model(X_test_tensor).numpy())
            
    return losses, predictions

def train_with_ntk(model, X_tensor, y_tensor, X_test_tensor, epochs=100, lr=0.01):
    """Train model using NTK dynamics with improved numerical stability"""
    loss_fn = nn.MSELoss()
    losses = []
    predictions = []
    width = getattr(model, 'width', 128)

    
    # Initial prediction
    with torch.no_grad():
        predictions.append(model(X_test_tensor).cpu().numpy())
    
    # Process in chunks for large datasets
    chunk_size = min(X_tensor.shape[0], 1000)
    
    for epoch in tqdm(range(epochs), desc="NTK Training"):
        with torch.no_grad():
            current_outputs = model(X_tensor)
            current_loss = loss_fn(current_outputs, y_tensor).item()
            losses.append(current_loss)
        
        for param in model.parameters():
            param.requires_grad_(True)
        
        # Process in chunks to avoid memory issues
        all_updates = torch.zeros_like(current_outputs)
        
        for chunk_start in range(0, len(X_tensor), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(X_tensor))
            chunk_indices = slice(chunk_start, chunk_end)
            
            # Compute NTK for this chunk
            ntk_chunk = compute_ntk(model, X_tensor[chunk_indices], X_tensor)
            
            # Compute update for this chunk
            chunk_update = -lr * torch.matmul(ntk_chunk, (current_outputs - y_tensor))
            all_updates[chunk_indices] = chunk_update
        
        # Target function values after NTK update
        target_outputs = current_outputs + all_updates
        
        # Update parameters with adaptive fitting
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Adaptive fitting steps
        fitting_steps = min(10, max(3, int(5 * width/128)))
        
        for _ in range(fitting_steps):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            target_fitting_loss = loss_fn(outputs, target_outputs.detach())
            target_fitting_loss.backward()
            optimizer.step()
        
        # Record test prediction
        with torch.no_grad():
            predictions.append(model(X_test_tensor).cpu().numpy())
    
    return losses, predictions

def check_convergence(losses, patience=10, min_delta=1e-4, min_epochs=50):
    """Check if training has converged based on recent loss history"""
    if len(losses) < patience + 1:
        return False
        
    # Require minimum number of epochs
    if len(losses) < min_epochs:
        return False
        
    # Check if loss hasn't improved by min_delta for patience epochs
    recent_best = min(losses[-patience:])
    if recent_best > min(losses[:-patience]) - min_delta:
        return True
    
    return False
