import torch
import numpy as np

def compute_ntk_batched(model, x1, x2=None, batch_size=10):
    """Compute the NTK with efficient batching"""
    if x2 is None:
        x2 = x1
        
    n1, n2 = x1.shape[0], x2.shape[0]
    ntk_matrix = torch.zeros((n1, n2), device=x1.device)
    
    # Pre-compute and store gradients for x1
    grad_vectors_x1 = []
    
    for i in range(0, n1, batch_size):
        batch_end = min(i + batch_size, n1)
        batch_x1 = x1[i:batch_end].clone().requires_grad_(True)
        
        with torch.enable_grad():
            outputs = model(batch_x1)
            
            for j in range(batch_x1.size(0)):
                model.zero_grad(set_to_none=True)  # More efficient
                
                if outputs.shape[0] > 1:
                    output = outputs[j:j+1]
                else:
                    output = outputs
                    
                grads = []
                for param in model.parameters():
                    if param.requires_grad:
                        grad = torch.autograd.grad(output, param, retain_graph=True)[0]
                        grads.append(grad.detach().flatten())
                
                grad_vectors_x1.append(torch.cat(grads))
    
    grad_vectors_x1 = torch.stack(grad_vectors_x1)
    
    # Now compute NTK in batches
    for i in range(0, n2, batch_size):
        batch_end = min(i + batch_size, n2)
        batch_x2 = x2[i:batch_end].clone().requires_grad_(True)
        
        batch_grads_x2 = []
        with torch.enable_grad():
            outputs = model(batch_x2)
            
            for j in range(batch_x2.size(0)):
                model.zero_grad(set_to_none=True)
                
                if outputs.shape[0] > 1:
                    output = outputs[j:j+1]
                else:
                    output = outputs
                
                grads = []
                for param in model.parameters():
                    if param.requires_grad:
                        grad = torch.autograd.grad(output, param, retain_graph=True)[0]
                        grads.append(grad.detach().flatten())
                
                batch_grads_x2.append(torch.cat(grads))
        
        batch_grad_vectors_x2 = torch.stack(batch_grads_x2)
        
        # Compute inner products efficiently: (n1, d) x (batch_size, d)T = (n1, batch_size)
        ntk_batch = torch.matmul(grad_vectors_x1, batch_grad_vectors_x2.T)
        ntk_matrix[:, i:batch_end] = ntk_batch
    
    return ntk_matrix

def compute_ntk(model, x1, x2=None):
    """Improved NTK computation with adaptive batch size"""
    # Determine appropriate batch size based on model size
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    batch_size = max(1, min(10, 1000000 // max(param_count, 1)))
    
    return compute_ntk_batched(model, x1, x2, batch_size=batch_size)

# Add these at module level
global_max_sq_residual = 1.0  # Global cap
historical_thresholds = []    # Track threshold history

def compute_sntk(model, x, x_batch, residuals, epoch=0):
    """Balanced SNTK computation with selective stability mechanisms"""
    n_samples = x.shape[0]
    n_batch = x_batch.shape[0]
    
    # Calculate squared residuals
    residuals_np = residuals.detach().cpu().numpy()
    squared_residuals = residuals_np ** 2
    
    # Calculate statistics with higher threshold (2.5-sigma)
    mean_sq_res = np.mean(squared_residuals)
    std_sq_res = np.std(squared_residuals)
    
    # Start with a more permissive threshold
    base_threshold = mean_sq_res + 2.5 * std_sq_res
    
    # Decay factor that becomes less restrictive over time
    # Allowing more stochasticity as training stabilizes
    epoch_factor = min(1.0, 0.5 + (epoch / 400))
    
    # Apply an absolute cap that's high enough for stochasticity
    # but prevents complete divergence
    max_squared_residual = min(base_threshold * epoch_factor, 2.0)
    
    # Compute SNTK with balanced clipping
    K_x_batch = compute_ntk(model, x, x_batch)
    K_batch_x = K_x_batch.T
    
    # Process in chunks
    sntk_matrix = torch.zeros((n_samples, n_samples), device=x.device)
    chunk_size = min(100, n_samples)
    
    for chunk_start in range(0, n_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_samples)
        chunk_indices = slice(chunk_start, chunk_end)
        chunk_K_x_batch = K_x_batch[chunk_indices, :]
        
        for i in range(n_batch):
            # Simple hard clipping without additional dampening
            r_i_squared = min(residuals[i].item() ** 2, max_squared_residual)
            
            outer_product = torch.outer(chunk_K_x_batch[:, i], K_batch_x[i, :])
            sntk_matrix[chunk_indices, :] += r_i_squared * outer_product
    
    # Normalize and ensure symmetry
    sntk_matrix /= n_batch
    sntk_matrix = 0.5 * (sntk_matrix + sntk_matrix.T)
    
    # Add minimal regularization - just enough for numerical stability
    jitter = 1e-4 * torch.eye(sntk_matrix.shape[0], device=sntk_matrix.device)
    sntk_matrix += jitter
    
    return sntk_matrix

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error between two arrays"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))