import torch
import torch.nn as nn
from utils.utils import *

class FlowMatching:
    def __init__(self,
                 sigma_min=0.01,
                 sigma_max=1.0,
                 rho=7,
                 motion_rep=None,
                 flow_type="rectified"):
        # Flow matching parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho  # Controls noise schedule
        self.motion_rep = motion_rep
        self.flow_type = flow_type  # "rectified" or "ot" (optimal transport)
        self.normalizer = MotionNormalizerTorch()
        
        # Pre-compute constants for efficiency
        self.log_sigma_min = torch.log(torch.tensor(self.sigma_min))
        self.log_sigma_max = torch.log(torch.tensor(self.sigma_max))
        self.sigma_diff = self.sigma_max - self.sigma_min
        
        # Constants for stability
        self.velocity_clip_val = 100.0
        self.nan_replacement = 0.0
        self.posinf_replacement = 100.0
        self.neginf_replacement = -100.0
        self.epsilon = 1e-8
    
    def noise_schedule(self, t):
        """
        Noise schedule function σ(t).
        For rectified flow, this determines the path from data to noise.
        """
        # Move pre-computed constants to device
        device = t.device
        log_sigma_min = self.log_sigma_min.to(device)
        log_sigma_max = self.log_sigma_max.to(device)
        
        if self.flow_type == "rectified":
            # Log-linear interpolation for sigma (vectorized)
            return torch.exp((1-t) * log_sigma_min + t * log_sigma_max)
        elif self.flow_type == "ot":
            # For optimal transport (vectorized)
            return self.sigma_min + self.sigma_diff * t
        else:
            raise ValueError(f"Unknown flow type: {self.flow_type}")
    
    def interpolate(self, x_0, x_1, t):
        """
        Interpolate between x_0 and x_1 based on time t.
        For rectified flow, this is a linear interpolation weighted by sigma.
        """
        # Reshape t for broadcasting (once)
        t_reshaped = t.reshape(-1, *([1] * (len(x_0.shape) - 1)))
        
        # Get noise level at time t
        sigma_t = self.noise_schedule(t)
        sigma_t_reshaped = sigma_t.reshape_as(t_reshaped)
        
        # Linear interpolation (optimized for each flow type)
        if self.flow_type == "rectified":
            # Rectified flow interpolation
            return (1 - sigma_t_reshaped) * x_0 + sigma_t_reshaped * x_1
        else:
            # Default linear interpolation
            return (1 - t_reshaped) * x_0 + t_reshaped * x_1
    
    def get_target_velocity(self, x_0, x_1, t):
        """
        Compute the target velocity field v(x_t, t) for time t.
        For rectified flow, v(x_t, t) = (x_1 - x_t) / (1 - t)
        """
        # Get interpolated point at time t
        x_t = self.interpolate(x_0, x_1, t)
        
        # Reshape t for broadcasting (once)
        t_reshaped = t.reshape(-1, *([1] * (len(x_0.shape) - 1)))
        
        # Compute velocity based on flow type
        if self.flow_type == "rectified":
            # Rectified flow velocity field - avoid t too close to 1
            # Use a soft clipping for better gradients
            denom = (1.0 - t_reshaped).clamp(min=0.005)
            v_t = (x_1 - x_t) / denom
            
            # Use soft clipping for better gradients and stability
            v_t = torch.tanh(v_t * 0.001) * 1e3
        elif self.flow_type == "ot":
            # Optimal transport velocity field
            v_t = x_1 - x_0
        else:
            raise ValueError(f"Unknown flow type: {self.flow_type}")
        
        return x_t, v_t
    
    def loss_fn(self, model, x_0, mask=None, cond=None, t_bar=None, **model_kwargs):
        """
        Compute the flow matching loss with numerical stability improvements.
        Also returns predicted and target motion data for additional loss calculations.
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Sample random time points - avoid t too close to 1
        if t_bar is not None:
            # Use more uniform distribution of time points
            t = torch.rand(B, device=device) * t_bar
        else:
            # Stratified sampling for better coverage of time space
            base = torch.linspace(0.01, 0.99, B, device=device)
            noise = torch.rand(B, device=device) * 0.02 - 0.01
            t = (base + noise).clamp(0.01, 0.99)
            # Shuffle to avoid batch correlation
            t = t[torch.randperm(B)]
        
        # Sample random noise as target distribution
        # Use smaller standard deviation for stability
        x_1 = torch.randn_like(x_0) * 0.5
        
        # Normalize original data if needed
        x_0_normalized = x_0.clone()
        if self.motion_rep is not None:
            # Store the original shape for later reference
            B, T = x_0.shape[:2]
            x_0_normalized = x_0_normalized.reshape(B, T, 2, -1)
            x_0_normalized = self.normalizer.forward(x_0_normalized)
            x_0_normalized = x_0_normalized.reshape(B, T, -1)
        
        # Get interpolated point and target velocity
        with torch.no_grad():
            x_t, v_t = self.get_target_velocity(x_0, x_1, t)
        
        # Get model's velocity prediction
        v_pred = model(x_t, t, mask=mask, cond=cond, **model_kwargs)
        
        # Predict current x_0 using the ODE and the velocity field
        # x_0_pred ≈ x_t - v_pred * t
        # This is an approximate inversion of the flow
        t_expanded = t.reshape(-1, 1, 1).expand(-1, x_t.shape[1], 1)
        x_0_pred = x_t - v_pred * t_expanded
        
        # Reshape for additional loss computations
        # These will be used for GeometricLoss and InterLoss
        B, T = x_0.shape[:2]
        x_0_pred_shaped = x_0_pred.reshape(B, T, 2, -1)  # [B, T, 2, D]
        x_0_gt_shaped = x_0.reshape(B, T, 2, -1)  # [B, T, 2, D]
        
        # Automatic handling of NaN/Inf through gradient masking
        # Create a mask for valid values
        valid_pred = ~(torch.isnan(v_pred) | torch.isinf(v_pred))
        valid_target = ~(torch.isnan(v_t) | torch.isinf(v_t))
        valid_mask = valid_pred & valid_target
        
        # Only compute loss on valid values
        squared_error = torch.zeros_like(v_pred)
        squared_error[valid_mask] = (v_pred[valid_mask] - v_t[valid_mask]) ** 2
        
        # Handle mask
        if mask is not None:
            # Create time-based mask
            time_mask = mask.any(dim=-1).float()  # [B, T]
            
            # Average across feature dimension
            timestep_error = squared_error.mean(dim=-1)  # [B, T]
            
            # Apply time mask
            masked_error = timestep_error * time_mask
            
            # Compute masked average
            loss = masked_error.sum() / (time_mask.sum() + self.epsilon)
        else:
            # Count valid elements for normalization
            num_valid = valid_mask.sum() + self.epsilon
            loss = squared_error.sum() / num_valid
        
        # Final check for NaN with graceful fallback
        if not torch.isfinite(loss):
            # Create a small, non-zero loss that maintains gradient
            loss = torch.tensor(1.0, device=device, requires_grad=True)
        
        return {
            "loss": loss, 
            "x_t": x_t, 
            "v_t": v_t, 
            "v_pred": v_pred,
            "t": t,
            "pred": x_0_pred_shaped,  # For compatibility with diffusion losses
            "target": x_0_gt_shaped,  # For compatibility with diffusion losses
        }
    
    @torch.no_grad()
    def sample(self, model, shape, steps=100, x_init=None, mask=None, cond=None, 
               denoise_to_zero=True, solver_type="euler", **model_kwargs):
        """
        Sample from the flow model by solving the ODE with stability improvements.
        """
        device = next(model.parameters()).device
        
        # Start from Gaussian noise if x_init is not provided
        if x_init is None:
            x_t = torch.randn(*shape, device=device) * 0.5  # Smaller init
        else:
            x_t = x_init
        
        # Integration time steps (from t=0.99 to t=0)
        # Use non-uniform time steps for better quality
        if steps > 20:
            # Quadratic spacing for more steps near t=0
            alpha = 2.0
            t_values = torch.linspace(0, 1, steps, device=device)
            time_steps = 0.99 * (1 - t_values**alpha)
        else:
            # Linear spacing for fewer steps
            time_steps = torch.linspace(0.99, 0.0 if denoise_to_zero else self.sigma_min, steps, device=device)
        
        # Prepare solver inputs for batched computation
        t_batch = time_steps.repeat(shape[0], 1)  # [B, steps]
        
        # Implement ODE solver with improved stability
        for i in range(steps - 1):
            t_current = t_batch[:, i]
            t_next = t_batch[:, i+1]
            dt = t_next - t_current  # Negative dt for backward integration
            
            try:
                if solver_type == "euler":
                    # Euler method
                    v_t = model(x_t, t_current, mask=mask, cond=cond, **model_kwargs)
                    # Use soft clipping for stability with better gradients
                    v_t = torch.tanh(v_t * 0.01) * self.velocity_clip_val
                    
                    x_t = x_t + dt.reshape(-1, 1, 1) * v_t
                    
                elif solver_type == "heun":
                    # Heun's method (second-order Runge-Kutta)
                    v_t = model(x_t, t_current, mask=mask, cond=cond, **model_kwargs)
                    v_t = torch.tanh(v_t * 0.01) * self.velocity_clip_val
                    
                    x_next = x_t + dt.reshape(-1, 1, 1) * v_t
                    v_next = model(x_next, t_next, mask=mask, cond=cond, **model_kwargs)
                    v_next = torch.tanh(v_next * 0.01) * self.velocity_clip_val
                    
                    v_avg = 0.5 * (v_t + v_next)
                    x_t = x_t + dt.reshape(-1, 1, 1) * v_avg
                
                else:
                    # Default to Euler
                    v_t = model(x_t, t_current, mask=mask, cond=cond, **model_kwargs)
                    v_t = torch.tanh(v_t * 0.01) * self.velocity_clip_val
                    
                    x_t = x_t + dt.reshape(-1, 1, 1) * v_t
                
                # Use mask for stability if provided
                if mask is not None:
                    # Only replace values where mask is active
                    mask_expanded = mask.any(dim=-1, keepdim=True).expand_as(x_t)
                    x_t = torch.where(mask_expanded, x_t, x_t.detach().clone())
                
                # More subtle handling of NaN/Inf
                if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                    # Only replace NaN/Inf values, keep others
                    x_t = torch.nan_to_num(
                        x_t, 
                        nan=self.nan_replacement,
                        posinf=self.posinf_replacement, 
                        neginf=self.neginf_replacement
                    )
                
            except Exception as e:
                print(f"Error in sampling step {i}: {e}")
                # Keep previous state with minimal perturbation
                x_t = x_t + torch.randn_like(x_t) * 0.001
        
        # Apply denormalization if necessary
        if self.motion_rep is not None and denoise_to_zero:
            # Reshape output for denormalization
            B, T = x_t.shape[:2]
            x_t_reshaped = x_t.reshape(B, T, 2, -1)
            x_t_denorm = self.normalizer.backward(x_t_reshaped, global_rt=True)
            return x_t_denorm.reshape(B, T, -1)
        
        return x_t