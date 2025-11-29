# Copyright (c) 2023 Predibase, Inc., 2019 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example physics-informed loss functions for Ludwig Feature Tensors Support.

This module demonstrates how to implement custom physics-informed loss functions
that leverage input feature tensors to enforce domain constraints and physical laws.
"""

import torch
import torch.nn as nn
from typing import Optional

from ludwig.modules.loss_modules import BaseLoss, register_loss
from ludwig.schema.features.loss.loss import MSELossConfig
from ludwig.utils.loss_utils import FeatureTensorDict


# ============================================================================
# Physics-Informed Loss Functions
# ============================================================================

class ConservationLoss(BaseLoss):
    """Physics-informed loss enforcing conservation principles.
    
    This loss function ensures that the sum of inputs equals the sum of outputs,
    which is useful for problems involving conservation of mass, energy, or other quantities.
    
    Example use cases:
    - Chemical reaction modeling (mass conservation)
    - Energy balance problems 
    - Financial flow analysis
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.expects_feature_tensors = True
        self.mse = nn.MSELoss()
        
        # Configuration parameters
        self.conservation_weight = getattr(config, 'conservation_weight', 100.0)
        self.input_features = getattr(config, 'conservation_input_features', ['input_mass'])
        self.tolerance = getattr(config, 'conservation_tolerance', 1e-6)
        
    def forward(self, predictions, targets, feature_tensors: Optional[FeatureTensorDict] = None):
        # Standard data fitting loss
        data_loss = self.mse(predictions, targets)
        
        if feature_tensors is None:
            return data_loss
            
        # Conservation constraint: sum of inputs should equal sum of outputs
        conservation_penalty = 0.0
        input_sum = 0.0
        
        for feature_name in self.input_features:
            if feature_name in feature_tensors:
                input_sum += feature_tensors[feature_name].sum(dim=-1, keepdim=True)
                
        if input_sum.numel() > 0:
            output_sum = predictions.sum(dim=-1, keepdim=True)
            conservation_violation = torch.abs(input_sum - output_sum)
            conservation_penalty = self.conservation_weight * conservation_violation.mean()
        
        return data_loss + conservation_penalty


class MonotonicityLoss(BaseLoss):
    """Physics-informed loss enforcing monotonic relationships.
    
    Ensures that the output has a monotonic relationship with specified input features.
    Useful for problems where physical laws dictate that one variable should
    increase/decrease monotonically with another.
    
    Example use cases:
    - Temperature vs. reaction rate (Arrhenius relationship)
    - Pressure vs. flow rate relationships
    - Price-demand relationships in economics
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.expects_feature_tensors = True
        self.mse = nn.MSELoss()
        
        # Configuration parameters
        self.monotonic_weight = getattr(config, 'monotonic_weight', 50.0)
        self.monotonic_features = getattr(config, 'monotonic_features', {})  # {feature: 'increasing'/'decreasing'}
        
    def forward(self, predictions, targets, feature_tensors: Optional[FeatureTensorDict] = None):
        # Standard data fitting loss
        data_loss = self.mse(predictions, targets)
        
        if feature_tensors is None or not self.monotonic_features:
            return data_loss
            
        monotonic_penalty = 0.0
        
        for feature_name, direction in self.monotonic_features.items():
            if feature_name in feature_tensors:
                feature_values = feature_tensors[feature_name].squeeze()
                
                # Sort by feature values to check monotonicity
                sorted_indices = torch.argsort(feature_values)
                sorted_predictions = predictions.squeeze()[sorted_indices]
                
                # Compute differences between consecutive predictions
                pred_diffs = sorted_predictions[1:] - sorted_predictions[:-1]
                
                if direction == 'increasing':
                    # Penalize negative differences (decreasing predictions)
                    violations = torch.relu(-pred_diffs)
                elif direction == 'decreasing':
                    # Penalize positive differences (increasing predictions)
                    violations = torch.relu(pred_diffs)
                else:
                    continue
                    
                monotonic_penalty += self.monotonic_weight * violations.mean()
        
        return data_loss + monotonic_penalty


class BoundaryConditionLoss(BaseLoss):
    """Physics-informed loss enforcing boundary conditions.
    
    Ensures that predictions satisfy specified boundary conditions based on input features.
    Common in PDE solving where certain boundary values must be maintained.
    
    Example use cases:
    - Heat transfer problems (fixed temperature boundaries)
    - Fluid dynamics (no-slip boundary conditions)
    - Structural analysis (fixed support conditions)
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.expects_feature_tensors = True
        self.mse = nn.MSELoss()
        
        # Configuration parameters
        self.boundary_weight = getattr(config, 'boundary_weight', 200.0)
        self.boundary_conditions = getattr(config, 'boundary_conditions', {})
        # Format: {feature_name: {'min_value': val, 'max_value': val, 'boundary_value': val}}
        
    def forward(self, predictions, targets, feature_tensors: Optional[FeatureTensorDict] = None):
        # Standard data fitting loss
        data_loss = self.mse(predictions, targets)
        
        if feature_tensors is None or not self.boundary_conditions:
            return data_loss
            
        boundary_penalty = 0.0
        
        for feature_name, conditions in self.boundary_conditions.items():
            if feature_name not in feature_tensors:
                continue
                
            feature_values = feature_tensors[feature_name]
            
            # Check for boundary conditions
            if 'min_value' in conditions and 'boundary_value' in conditions:
                min_val = conditions['min_value']
                boundary_val = conditions['boundary_value']
                
                # Find points near the minimum boundary
                boundary_mask = (feature_values <= min_val + 0.1).float()
                if boundary_mask.sum() > 0:
                    boundary_predictions = predictions * boundary_mask
                    target_boundary = torch.full_like(boundary_predictions, boundary_val)
                    boundary_penalty += self.boundary_weight * self.mse(
                        boundary_predictions, target_boundary * boundary_mask
                    )
            
            if 'max_value' in conditions and 'boundary_value' in conditions:
                max_val = conditions['max_value']
                boundary_val = conditions['boundary_value']
                
                # Find points near the maximum boundary  
                boundary_mask = (feature_values >= max_val - 0.1).float()
                if boundary_mask.sum() > 0:
                    boundary_predictions = predictions * boundary_mask
                    target_boundary = torch.full_like(boundary_predictions, boundary_val)
                    boundary_penalty += self.boundary_weight * self.mse(
                        boundary_predictions, target_boundary * boundary_mask
                    )
        
        return data_loss + boundary_penalty


class PhysicalRangeLoss(BaseLoss):
    """Physics-informed loss enforcing physically realistic output ranges.
    
    Constrains predictions to physically meaningful ranges based on input context.
    Helps prevent models from making unrealistic predictions that violate physical limits.
    
    Example use cases:
    - Temperature predictions (cannot go below absolute zero)
    - Probability outputs (must be between 0 and 1)
    - Concentration values (non-negative, below saturation limits)
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.expects_feature_tensors = True
        self.mse = nn.MSELoss()
        
        # Configuration parameters
        self.range_weight = getattr(config, 'range_weight', 100.0)
        self.absolute_min = getattr(config, 'absolute_min', None)
        self.absolute_max = getattr(config, 'absolute_max', None)
        self.context_ranges = getattr(config, 'context_ranges', {})
        # Format: {feature_name: {'feature_min': val, 'feature_max': val, 'output_min': val, 'output_max': val}}
        
    def forward(self, predictions, targets, feature_tensors: Optional[FeatureTensorDict] = None):
        # Standard data fitting loss
        data_loss = self.mse(predictions, targets)
        range_penalty = 0.0
        
        # Absolute range constraints
        if self.absolute_min is not None:
            violations = torch.relu(self.absolute_min - predictions)
            range_penalty += self.range_weight * violations.mean()
            
        if self.absolute_max is not None:
            violations = torch.relu(predictions - self.absolute_max)
            range_penalty += self.range_weight * violations.mean()
            
        # Context-dependent range constraints
        if feature_tensors is not None and self.context_ranges:
            for feature_name, ranges in self.context_ranges.items():
                if feature_name not in feature_tensors:
                    continue
                    
                feature_values = feature_tensors[feature_name]
                
                # Define context-dependent ranges
                feature_min = ranges.get('feature_min', float('-inf'))
                feature_max = ranges.get('feature_max', float('inf'))
                output_min = ranges.get('output_min', float('-inf'))
                output_max = ranges.get('output_max', float('inf'))
                
                # Apply constraints when feature is in specified range
                context_mask = ((feature_values >= feature_min) & 
                              (feature_values <= feature_max)).float()
                
                if context_mask.sum() > 0:
                    # Minimum constraint
                    if output_min != float('-inf'):
                        violations = torch.relu(output_min - predictions) * context_mask
                        range_penalty += self.range_weight * violations.mean()
                        
                    # Maximum constraint
                    if output_max != float('inf'):
                        violations = torch.relu(predictions - output_max) * context_mask
                        range_penalty += self.range_weight * violations.mean()
        
        return data_loss + range_penalty


class DifferentialEquationLoss(BaseLoss):
    """Physics-informed loss enforcing differential equation constraints.
    
    Implements basic differential equation constraints for problems where
    the relationship between inputs and outputs follows known differential equations.
    
    Example use cases:
    - Population dynamics (exponential/logistic growth)
    - Radioactive decay
    - Simple harmonic motion
    - Economic growth models
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.expects_feature_tensors = True
        self.mse = nn.MSELoss()
        
        # Configuration parameters
        self.de_weight = getattr(config, 'differential_weight', 50.0)
        self.equation_type = getattr(config, 'equation_type', 'exponential')
        self.time_feature = getattr(config, 'time_feature', 'time')
        self.rate_feature = getattr(config, 'rate_feature', 'rate')
        
    def forward(self, predictions, targets, feature_tensors: Optional[FeatureTensorDict] = None):
        # Standard data fitting loss
        data_loss = self.mse(predictions, targets)
        
        if feature_tensors is None:
            return data_loss
            
        de_penalty = 0.0
        
        if self.equation_type == 'exponential' and self.time_feature in feature_tensors:
            # For exponential growth: dy/dt = k*y
            time_values = feature_tensors[self.time_feature]
            
            if self.rate_feature in feature_tensors:
                rate_values = feature_tensors[self.rate_feature]
                
                # Approximate derivative using finite differences
                if time_values.shape[0] > 1:
                    # Sort by time for proper derivative calculation
                    sorted_indices = torch.argsort(time_values.squeeze())
                    sorted_times = time_values[sorted_indices]
                    sorted_predictions = predictions[sorted_indices]
                    sorted_rates = rate_values[sorted_indices]
                    
                    # Calculate derivatives
                    dt = sorted_times[1:] - sorted_times[:-1]
                    dy = sorted_predictions[1:] - sorted_predictions[:-1]
                    dy_dt = dy / (dt + 1e-8)  # Add small epsilon to avoid division by zero
                    
                    # Expected derivative from exponential law
                    expected_dy_dt = sorted_rates[1:] * sorted_predictions[1:]
                    
                    # Penalize deviation from exponential relationship
                    de_violation = torch.abs(dy_dt - expected_dy_dt)
                    de_penalty = self.de_weight * de_violation.mean()
        
        return data_loss + de_penalty


# ============================================================================
# Register Custom Loss Functions
# ============================================================================

# Register the physics-informed losses for use in Ludwig configurations
register_loss(MSELossConfig)(ConservationLoss)
register_loss(MSELossConfig)(MonotonicityLoss) 
register_loss(MSELossConfig)(BoundaryConditionLoss)
register_loss(MSELossConfig)(PhysicalRangeLoss)
register_loss(MSELossConfig)(DifferentialEquationLoss)