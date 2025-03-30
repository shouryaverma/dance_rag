import torch
import numpy as np
from tqdm.auto import tqdm
from enum import Enum
from utils.utils import MotionNormalizerTorch
from models.losses import InterLoss, GeometricLoss

class FlowType(Enum):
    """
    Types of flow implementations
    """
    RECTIFIED = "rectified"
    LINEAR = "linear"

class RectifiedFlow:
    """
    Implementation of Rectified Flow with discrete timesteps.
    """
    
    def __init__(
        self, 
        num_timesteps=1000, 
        flow_type=FlowType.RECTIFIED,
        rescale_timesteps=False,
        motion_rep="global"
    ):
        self.num_timesteps = num_timesteps
        self.flow_type = flow_type
        self.rescale_timesteps = rescale_timesteps
        self.motion_rep = motion_rep
        self.normalizer = MotionNormalizerTorch()
        
        # Set up timesteps
        self.timesteps = torch.linspace(0, 1, num_timesteps)
        
    def interpolate(self, x0, x1, t):
        """
        Interpolate between x0 and x1 at time t.
        For rectified flow, this is a straight line path.
        """
        t = t.view(-1, 1, 1)  # Reshape for broadcasting
        return (1 - t) * x0 + t * x1
    
    def compute_vector_field(self, x0, x1, t):
        """
        Compute the vector field (velocity) for rectified flow.
        For rectified flow, the velocity field is a constant (x1 - x0)
        """
        return x1 - x0
    
    def sample_noise(self, shape, device):
        """
        Sample random noise with the same shape as the data.
        """
        return torch.randn(shape, device=device)
    
    def forward_process(self, x_start, t_normalized):
        """
        Forward process: interpolate from data to noise at time t.
        """
        # Generate noise
        noise = self.sample_noise(x_start.shape, x_start.device)
        
        # Interpolate
        x_t = self.interpolate(x_start, noise, t_normalized)
        
        return x_t, noise
    
    def compute_loss(self, model, x_start, t, mask=None, timestep_mask=None, t_bar=None, model_kwargs=None):
        """
        Compute the loss for training the flow model.
        """
        if model_kwargs is None:
            model_kwargs = {}
            
        # Get batch and sequence dimensions
        B, T = x_start.shape[:2]
        
        # Normalize timesteps to [0, 1]
        t_normalized = t.float() / self.num_timesteps
        
        # Reshape the data for duet format (2 dancers)
        x_start_shaped = x_start.reshape(B, T, 2, -1)
        
        # Important: Reshape mask to match expected dimensions
        if mask is not None:
            mask = mask.reshape(B, T, -1, 1)
        
        # Apply normalization as in the original code
        if self.motion_rep == "global":
            x_start_normalized = self.normalizer.forward(x_start_shaped)
            # Reshape back to concatenated format
            x_start_normalized = x_start_normalized.reshape(B, T, -1)
        else:
            x_start_normalized = x_start
        
        # Forward process on normalized data
        x_t, noise = self.forward_process(x_start_normalized, t_normalized)
        
        # True vector field (velocity)
        true_velocity = self.compute_vector_field(x_start_normalized, noise, t_normalized)
        
        # Predicted velocity
        pred_velocity = model(x_t, self._scale_timesteps(t), **model_kwargs)
        
        # Initialize loss dictionary
        losses = {}
        
        # First compute simple MSE loss
        simple_loss = ((true_velocity - pred_velocity) ** 2).mean()
        losses["simple"] = simple_loss
        
        # Apply specialized motion losses if mask is provided and we're using global representation
        if mask is not None and self.motion_rep == "global":
            # Reshape predicted and target velocities for the motion losses
            prediction = pred_velocity.reshape(B, T, 2, -1)
            target = true_velocity.reshape(B, T, 2, -1)
            
            # Calculate inter-dancer losses
            interloss_manager = InterLoss("l2", 22)
            interloss_manager.forward(prediction, target, mask, timestep_mask)
            
            # Calculate per-dancer losses
            loss_a_manager = GeometricLoss("l2", 22, "A")
            loss_a_manager.forward(prediction[...,0,:], target[...,0,:], mask[...,0,:], timestep_mask)
            
            loss_b_manager = GeometricLoss("l2", 22, "B")
            loss_b_manager.forward(prediction[...,1,:], target[...,1,:], mask[...,0,:], timestep_mask)
            
            # Combine all losses
            losses.update(loss_a_manager.losses)
            losses.update(loss_b_manager.losses)
            losses.update(interloss_manager.losses)
            losses["total"] = loss_a_manager.losses["A"] + loss_b_manager.losses["B"] + interloss_manager.losses["total"]
        else:
            losses["total"] = simple_loss
        
        return losses
    
    def _euler_step(self, x, velocity, dt):
        """Helper function for Euler integration step"""
        return x - velocity * dt
    
    def _heun_step(self, model, x, t, t_next, dt, model_kwargs):
        """
        Heun's method (improved Euler) for more accurate integration
        """
        velocity_t = model(x, self._scale_timesteps(t), **model_kwargs)
        x_euler = self._euler_step(x, velocity_t, dt)
        
        velocity_t_next = model(x_euler, self._scale_timesteps(t_next), **model_kwargs)
        velocity_avg = 0.5 * (velocity_t + velocity_t_next)
        
        return self._euler_step(x, velocity_avg, dt)
    
    def sample(self, model, shape, noise=None, clip_denoised=True, model_kwargs=None, device=None, progress=False):
        """
        Sample from the model using numerical integration of the flow ODE.
        Uses Heun's method for better numerical stability.
        """
        if device is None:
            device = next(model.parameters()).device
            
        if noise is None:
            noise = self.sample_noise(shape, device)
        
        # Start from noise
        x = noise
        
        # Timesteps for integration (backward from t=1 to t=0)
        original_timesteps = list(range(self.num_timesteps))[::-1]
        
        # Keep original_timesteps as a list for indexing, but use tqdm for display if needed
        timesteps_iter = tqdm(original_timesteps) if progress else original_timesteps
        
        # Improved numerical integration with Heun's method
        for i, t in enumerate(timesteps_iter):
            t_tensor = torch.tensor([t] * shape[0], device=device)
            
            # For the last step, use Euler
            if i == len(original_timesteps) - 1:
                with torch.no_grad():
                    velocity = model(x, self._scale_timesteps(t_tensor), **model_kwargs)
                dt = 1.0 / self.num_timesteps
                x = self._euler_step(x, velocity, dt)
            else:
                # Get next timestep from the original list, not from tqdm wrapper
                t_next = original_timesteps[i + 1]
                t_next_tensor = torch.tensor([t_next] * shape[0], device=device)
                dt = 1.0 / self.num_timesteps
                
                with torch.no_grad():
                    x = self._heun_step(model, x, t_tensor, t_next_tensor, dt, model_kwargs)
                
        return x
    
    def _scale_timesteps(self, t):
        """
        Scale timesteps if needed.
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t