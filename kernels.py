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
    """Enhanced SNTK computation with robust stability mechanisms"""
    global global_max_sq_residual, historical_thresholds
    
    n_samples = x.shape[0]
    n_batch = x_batch.shape[0]
    
    # Step 1: Calculate baseline statistical threshold
    residuals_np = residuals.detach().cpu().numpy()
    squared_residuals = residuals_np ** 2
    
    # Step 2: More conservative sigma (2-sigma instead of 3-sigma)
    mean_sq_res = np.mean(squared_residuals)
    std_sq_res = np.std(squared_residuals)
    stat_threshold = mean_sq_res + 2.0 * std_sq_res
    
    # Step 3: Apply global cap with decay
    global_max_sq_residual = min(global_max_sq_residual, max(0.5, 10.0 / (epoch + 10)))
    
    # Step 4: Percentile-based threshold as backup strategy
    percentile_threshold = np.percentile(squared_residuals, 95)
    
    # Step 5: Use minimum of all threshold strategies
    max_squared_residual = min(
        stat_threshold,
        global_max_sq_residual, 
        percentile_threshold,
        0.5  # Hard maximum cap
    )
    
    # Step 6: Ensure minimum floor for early training stability
    if epoch < 10:
        max_squared_residual = max(max_squared_residual, 0.05)
    
    # Step 7: Track threshold history for monitoring
    historical_thresholds.append(max_squared_residual)
    
    # Step 8: Apply temporal smoothing if we have history
    if len(historical_thresholds) > 5:
        max_squared_residual = 0.7 * max_squared_residual + 0.3 * np.mean(historical_thresholds[-5:])
    
    # Compute SNTK with robust clipping
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
            # Apply enhanced clipping
            r_i_squared = min(residuals[i].item() ** 2, max_squared_residual)
            
            # Step 9: Apply soft clipping through logarithmic dampening for large values
            if r_i_squared > max_squared_residual * 0.5:
                dampen_factor = 0.5 + 0.5 * (max_squared_residual * 0.5) / max(r_i_squared, 1e-10)
                r_i_squared *= dampen_factor
                
            outer_product = torch.outer(chunk_K_x_batch[:, i], K_batch_x[i, :])
            sntk_matrix[chunk_indices, :] += r_i_squared * outer_product
    
    # Normalize and ensure symmetry
    sntk_matrix /= n_batch
    sntk_matrix = 0.5 * (sntk_matrix + sntk_matrix.T)
    
    # Step 10: Add adaptive regularization based on eigenvalue conditioning
    jitter = 1e-4 * (1.0 + np.mean(squared_residuals) * 10)
    sntk_matrix += jitter * torch.eye(sntk_matrix.shape[0], device=sntk_matrix.device)
    
    return sntk_matrix

def determine_max_squared_residual(residuals):
    """Determine maximum squared residual threshold using statistical analysis"""
    # Convert to numpy for statistical calculations
    residuals_np = residuals.detach().cpu().numpy()
    squared_residuals = residuals_np ** 2
    
    # Calculate statistics
    mean_sq_res = np.mean(squared_residuals)
    std_sq_res = np.std(squared_residuals)
    
    # Use 3-sigma rule for clipping threshold
    max_squared_residual = mean_sq_res + 3 * std_sq_res
    
    # Add minimal threshold to handle early training
    max_squared_residual = max(max_squared_residual, 0.1)
    
    return max_squared_residual

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error between two arrays"""
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))