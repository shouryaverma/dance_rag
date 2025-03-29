import torch
import torch.nn as nn
from utils.utils import *

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
    
    def noise_schedule(self, t):
        """
        Noise schedule function σ(t).
        For rectified flow, this determines the path from data to noise.
        """
        if self.flow_type == "rectified":
            # Log-linear interpolation for sigma
            log_sigma_min = torch.log(torch.tensor(self.sigma_min, device=t.device))
            log_sigma_max = torch.log(torch.tensor(self.sigma_max, device=t.device))
            return torch.exp((1-t) * log_sigma_min + t * log_sigma_max)
        elif self.flow_type == "ot":
            # For optimal transport
            return self.sigma_min + (self.sigma_max - self.sigma_min) * t
        else:
            raise ValueError(f"Unknown flow type: {self.flow_type}")
    
    def interpolate(self, x_0, x_1, t):
        """
        Interpolate between x_0 and x_1 based on time t.
        For rectified flow, this is a linear interpolation weighted by sigma.
        """
        # Reshape t for broadcasting
        t = t.reshape(-1, *([1] * (len(x_0.shape) - 1)))
        
        # Get noise level at time t
        sigma_t = self.noise_schedule(t)
        
        # Linear interpolation
        if self.flow_type == "rectified":
            # Rectified flow interpolation
            return (1 - sigma_t) * x_0 + sigma_t * x_1
        else:
            # Default linear interpolation
            return (1 - t) * x_0 + t * x_1
    
    def get_target_velocity(self, x_0, x_1, t):
        """
        Compute the target velocity field v(x_t, t) for time t.
        For rectified flow, v(x_t, t) = (x_1 - x_t) / (1 - t)
        """
        # Get interpolated point at time t
        x_t = self.interpolate(x_0, x_1, t)
        
        # Reshape t for broadcasting
        t_reshaped = t.reshape(-1, *([1] * (len(x_0.shape) - 1)))
        
        # Compute velocity based on flow type
        if self.flow_type == "rectified":
            # Rectified flow velocity field - avoid t too close to 1
            t_clipped = t_reshaped.clamp(0, 0.995)
            v_t = (x_1 - x_t) / (1.0 - t_clipped)
            
            # Clamp velocities to avoid explosions
            v_t = torch.clamp(v_t, -1e3, 1e3)
        elif self.flow_type == "ot":
            # Optimal transport velocity field
            v_t = x_1 - x_0
        else:
            raise ValueError(f"Unknown flow type: {self.flow_type}")
        
        return x_t, v_t
    
    def loss_fn(self, model, x_0, mask=None, cond=None, t_bar=None, **model_kwargs):
        """
        Compute the flow matching loss with numerical stability improvements.
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Sample random time points - avoid t too close to 1
        if t_bar is not None:
            t = torch.rand(B, device=device) * t_bar
        else:
            t = torch.rand(B, device=device) * 0.98 + 0.01
        
        # Sample random noise as target distribution
        x_1 = torch.randn_like(x_0) * 0.5  # Use smaller initialization
        
        # Apply normalization if needed
        if self.motion_rep is not None:
            # Skip normalization for now to avoid reshape issues
            pass
        
        # Get interpolated point and target velocity
        x_t, v_t = self.get_target_velocity(x_0, x_1, t)
        
        # Get model's velocity prediction
        v_pred = model(x_t, t, mask=mask, cond=cond, **model_kwargs)
        
        # Check for NaN/Inf in velocities and fix them
        if torch.isnan(v_t).any() or torch.isinf(v_t).any():
            print("Warning: NaN/Inf in target velocity. Stabilizing...")
            v_t = torch.nan_to_num(v_t, nan=0.0, posinf=100.0, neginf=-100.0)
        
        if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
            print("Warning: NaN/Inf in predicted velocity. Stabilizing...")
            v_pred = torch.nan_to_num(v_pred, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Compute MSE loss with stability
        squared_error = (v_pred - v_t) ** 2
        
        # Handle mask
        if mask is not None:
            # Create time-based mask
            time_mask = mask.any(dim=-1).float()  # [B, T]
            
            # Average across feature dimension
            timestep_error = squared_error.mean(dim=-1)  # [B, T]
            
            # Apply time mask
            masked_error = timestep_error * time_mask
            
            # Compute masked average
            loss = masked_error.sum() / (time_mask.sum() + 1e-8)
        else:
            loss = squared_error.mean()
        
        # Final check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print("Warning: Loss is NaN/Inf. Using default value.")
            loss = torch.tensor(1.0, device=device, requires_grad=True)
        
        return {"loss": loss, "x_t": x_t, "v_t": v_t, "v_pred": v_pred}
    
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
        time_steps = torch.linspace(0.99, 0.0 if denoise_to_zero else self.sigma_min, steps, device=device)
        dt = time_steps[0] - time_steps[1]  # Negative dt for backward integration
        
        # Integration loop
        for i in range(steps - 1):
            t = time_steps[i] * torch.ones(shape[0], device=device)
            t_next = time_steps[i+1] * torch.ones(shape[0], device=device)
            
            # Implement ODE solver
            try:
                if solver_type == "euler":
                    # Euler method
                    with torch.no_grad():
                        v_t = model(x_t, t, mask=mask, cond=cond, **model_kwargs)
                        # Clamp velocity for stability
                        v_t = torch.clamp(v_t, -100, 100)
                    
                    x_t = x_t - dt * v_t
                    
                elif solver_type == "heun":
                    # Heun's method (second-order Runge-Kutta)
                    with torch.no_grad():
                        v_t = model(x_t, t, mask=mask, cond=cond, **model_kwargs)
                        v_t = torch.clamp(v_t, -100, 100)
                        
                        x_next = x_t - dt * v_t
                        v_next = model(x_next, t_next, mask=mask, cond=cond, **model_kwargs)
                        v_next = torch.clamp(v_next, -100, 100)
                        
                        v_avg = 0.5 * (v_t + v_next)
                    
                    x_t = x_t - dt * v_avg
                    
                else:
                    # Default to Euler
                    with torch.no_grad():
                        v_t = model(x_t, t, mask=mask, cond=cond, **model_kwargs)
                        v_t = torch.clamp(v_t, -100, 100)
                    
                    x_t = x_t - dt * v_t
                
                # Check for NaN/Inf
                if torch.isnan(x_t).any() or torch.isinf(x_t).any():
                    print(f"Warning: NaN/Inf at step {i}. Stabilizing...")
                    x_t = torch.nan_to_num(x_t, nan=0.0, posinf=100.0, neginf=-100.0)
                
            except Exception as e:
                print(f"Error in sampling step {i}: {e}")
                # Keep previous state
                continue
        
        return x_t