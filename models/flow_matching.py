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

def mean_flat(x, temperature=1.0, **kwargs):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def class_conditioned_sampling(labels):
    bsz = labels.shape[0]
    mask = ~(labels[None] == labels[:, None])
    weights = mask.float()
    weights_sum = weights.sum(dim=1, keepdim=True)
    if (weights_sum == 0).any():
        choices = torch.randint(0, bsz, (bsz,), device=labels.device)
    else:
        weights = weights / weights_sum.clamp(min=1)
        choices = torch.multinomial(weights, 1).squeeze(1)
    assert (choices != torch.arange(bsz, device=labels.device)).all(), "Self-sampling detected"
    return choices

class RectifiedFlow:
    """
    Implementation of Rectified Flow with discrete timesteps.
    """
    
    def __init__(
        self, 
        num_timesteps=1000, 
        flow_type=FlowType.RECTIFIED,
        rescale_timesteps=False,
        motion_rep="global",

        use_contrastive=True,
        contrastive_weight=0.05,
        null_class_idx=None,
    ):
        self.num_timesteps = num_timesteps
        self.flow_type = flow_type
        self.rescale_timesteps = rescale_timesteps
        self.motion_rep = motion_rep
        self.normalizer = MotionNormalizerTorch()
        
        # Set up timesteps
        self.timesteps = torch.linspace(0, 1, num_timesteps)

        # Contrastive loss parameters
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.null_class_idx = null_class_idx

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
    
    def get_interpolant_derivatives(self, t):
        """
        Get derivatives for rectified flow interpolation.
        For rectified flow: x_t = (1-t)*x0 + t*x1
        So: d_alpha_t = -1, d_sigma_t = 1
        """
        d_alpha_t = -torch.ones_like(t)
        d_sigma_t = torch.ones_like(t)
        return d_alpha_t, d_sigma_t
    
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

    def compute_triplet_loss_efficiently(self, x, y, labels=None):
        """
        Compute triplet contrastive loss efficiently.
        """
        x = x.flatten(1)
        y = y.flatten(1)
        
        # Normalize to unit scale to prevent explosion
        x_norm = torch.nn.functional.normalize(x, dim=-1)
        y_norm = torch.nn.functional.normalize(y, dim=-1)
        
        # Positive error (cosine distance)
        pos_error = 1 - torch.sum(x_norm * y_norm, dim=-1)  # Shape: (B,)
        
        bsz = x.shape[0]
        if bsz < 2:
            # Skip contrastive loss if batch size too small
            return {
                "contrastive_loss": torch.tensor(0.0, device=x.device),
                "flow_loss": pos_error.mean(),
                "negative_loss": torch.tensor(0.0, device=x.device)
            }
            
        # Simpler negative sampling
        choices = torch.randperm(bsz, device=x.device)
        choices = torch.roll(choices, 1)  # Shift by 1 to avoid self-pairing
        
        y_neg_norm = y_norm[choices]
        
        # Negative error (cosine distance)
        neg_error = 1 - torch.sum(x_norm * y_neg_norm, dim=-1)  # Shape: (B,)
        
        # Triplet loss with margin
        margin = 0.1  # Small margin for stability
        triplet_loss = torch.clamp(pos_error - neg_error + margin, min=0.0)
        contrastive_loss = triplet_loss.mean()

        return {
            "contrastive_loss": contrastive_loss,
            "flow_loss": pos_error.mean(),
            "negative_loss": neg_error.mean()
        }
    
    def compute_contrastive_loss(self, pred_velocity, target_data, noise, t_normalized, labels=None):
        """
        Compute contrastive loss for rectified flow.
        """
        # Get interpolant derivatives for rectified flow
        d_alpha_t, d_sigma_t = self.get_interpolant_derivatives(t_normalized)
        d_alpha_t = d_alpha_t.view(-1, 1, 1)
        d_sigma_t = d_sigma_t.view(-1, 1, 1)
        
        # For rectified flow, the target velocity is x1 - x0 = noise - target_data
        # The model target in velocity space is: d_alpha_t * target_data + d_sigma_t * noise
        # For rectified flow: d_alpha_t = -1, d_sigma_t = 1
        # So model_target = -target_data + noise = noise - target_data (which is the true velocity)
        model_target = d_alpha_t * target_data + d_sigma_t * noise
        
        return self.compute_triplet_loss_efficiently(pred_velocity, model_target, labels)
    
    def compute_loss(self, model, x_start, t, mask=None, timestep_mask=None, t_bar=None, mode="duet", model_kwargs=None, labels=None):
        """
        Compute the loss for training the flow model with optional contrastive loss.
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
        primary_loss = simple_loss

        # Add contrastive loss if enabled
        if self.use_contrastive:
            contrastive_losses = self.compute_contrastive_loss(
                pred_velocity, x_start_normalized, noise, t_normalized, labels
            )
            losses.update(contrastive_losses)
            # Add contrastive to base MSE loss instead of replacing it
            weighted_contrastive = self.contrastive_weight * contrastive_losses["contrastive_loss"]
            primary_loss = simple_loss + weighted_contrastive
        
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
            
            # Combine motion losses
            motion_losses = {}
            motion_losses.update(loss_a_manager.losses)
            motion_losses.update(loss_b_manager.losses)
            motion_losses.update(interloss_manager.losses)
            
            if mode=="duet":
                motion_total = loss_a_manager.losses["A"] + loss_b_manager.losses["B"] + interloss_manager.losses["total"]
            else:
                motion_total = loss_a_manager.losses["A"]*0.001 + loss_b_manager.losses["B"]*1.0 + interloss_manager.losses["total"]
            
            # Combine contrastive and motion losses
            if self.use_contrastive:
                losses["motion_total"] = motion_total
                losses["total"] = primary_loss + motion_total
            else:
                losses.update(motion_losses)
                losses["total"] = motion_total
        else:
            losses["total"] = primary_loss
        
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
    
    def _dopri5_step(self, model, x, t, t_next, dt, model_kwargs):
        """
        Dormand-Prince (DOPRI5) method for more accurate integration
        This is a simplified implementation without adaptive step size
        """
        # DOPRI5 coefficients
        c2, c3, c4, c5, c6 = 1/5, 3/10, 4/5, 8/9, 1.0
        
        a21 = 1/5
        a31, a32 = 3/40, 9/40
        a41, a42, a43 = 44/45, -56/15, 32/9
        a51, a52, a53, a54 = 19372/6561, -25360/2187, 64448/6561, -212/729
        a61, a62, a63, a64, a65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656
        
        b1, b3, b4, b5, b6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84
        
        # Convert t and t_next to integers
        t_int = t.long()
        t_next_int = t_next.long()
        
        # Stage 1 - at time t
        k1 = model(x, self._scale_timesteps(t_int), **model_kwargs)
        
        # Stage 2
        # Instead of interpolating the timestep, we'll use discrete timesteps that are valid indices
        t2_float = t * (1 - c2) + t_next * c2
        t2_int = torch.clamp(t2_float.round().long(), min=0, max=self.num_timesteps-1)
        x2 = x - dt * a21 * k1
        k2 = model(x2, self._scale_timesteps(t2_int), **model_kwargs)
        
        # Stage 3
        t3_float = t * (1 - c3) + t_next * c3
        t3_int = torch.clamp(t3_float.round().long(), min=0, max=self.num_timesteps-1)
        x3 = x - dt * (a31 * k1 + a32 * k2)
        k3 = model(x3, self._scale_timesteps(t3_int), **model_kwargs)
        
        # Stage 4
        t4_float = t * (1 - c4) + t_next * c4
        t4_int = torch.clamp(t4_float.round().long(), min=0, max=self.num_timesteps-1)
        x4 = x - dt * (a41 * k1 + a42 * k2 + a43 * k3)
        k4 = model(x4, self._scale_timesteps(t4_int), **model_kwargs)
        
        # Stage 5
        t5_float = t * (1 - c5) + t_next * c5
        t5_int = torch.clamp(t5_float.round().long(), min=0, max=self.num_timesteps-1)
        x5 = x - dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4)
        k5 = model(x5, self._scale_timesteps(t5_int), **model_kwargs)
        
        # Stage 6
        # This is at t_next which is already an integer
        x6 = x - dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
        k6 = model(x6, self._scale_timesteps(t_next_int), **model_kwargs)
        
        # Combine results for the final step (5th order)
        x_next = x - dt * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
        
        return x_next
    
    def sample(self, model, shape, noise=None, model_kwargs=None, device=None, progress=False):
        """
        Sample from the model using numerical integration of the flow ODE.
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
        
        # Numerical integration with DOPRI5 method
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
                    x = self._dopri5_step(model, x, t_tensor, t_next_tensor, dt, model_kwargs)
                    # x = self._heun_step(model, x, t_tensor, t_next_tensor, dt, model_kwargs)
        return x
    
    def _scale_timesteps(self, t):
        """
        Scale timesteps if needed.
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t