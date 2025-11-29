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
"""Integration tests for end-to-end feature tensor training."""

import tempfile
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import torch

from ludwig.api import LudwigModel
from ludwig.modules.loss_modules import BaseLoss, register_loss
from ludwig.schema.features.loss.loss import MSELossConfig


# ============================================================================
# Test Physics-Informed Loss Functions
# ============================================================================

class SimplePhysicsLoss(BaseLoss):
    """Simple physics-informed loss for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self.expects_feature_tensors = True
        self.mse = torch.nn.MSELoss()
        
    def forward(self, predictions, targets, feature_tensors=None):
        data_loss = self.mse(predictions, targets)
        
        if feature_tensors is not None:
            # Simple constraint: penalize negative predictions when input_feature is positive
            input_feature = feature_tensors.get('input_feature')
            if input_feature is not None:
                positive_mask = (input_feature > 0).float()
                penalty = torch.relu(-predictions) * positive_mask
                physics_penalty = penalty.mean() * 100.0
                return data_loss + physics_penalty
        
        return data_loss


class ConstrainedOutputLoss(BaseLoss):
    """Physics loss that enforces output constraints."""
    
    def __init__(self, config):
        super().__init__(config)
        self.expects_feature_tensors = True
        self.mse = torch.nn.MSELoss()
        
    def forward(self, predictions, targets, feature_tensors=None):
        data_loss = self.mse(predictions, targets)
        
        if feature_tensors is not None:
            # Constraint: output should be less than max_input when constraint_feature is positive
            constraint_feature = feature_tensors.get('constraint_feature')
            max_input = feature_tensors.get('max_input')
            
            if constraint_feature is not None and max_input is not None:
                constraint_active = (constraint_feature > 0.5).float()
                violation = torch.relu(predictions - max_input) * constraint_active
                constraint_penalty = violation.mean() * 1000.0
                return data_loss + constraint_penalty
        
        return data_loss


# Register test losses
register_loss(MSELossConfig)(SimplePhysicsLoss)
register_loss(MSELossConfig)(ConstrainedOutputLoss)


# ============================================================================
# Integration Tests
# ============================================================================

class TestFeatureTensorIntegration:
    """End-to-end tests for feature tensor training."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        n_samples = 200
        
        # Generate synthetic data
        input_feature = np.random.randn(n_samples) * 2
        constraint_feature = np.random.choice([0, 1], n_samples)
        max_input = np.abs(np.random.randn(n_samples) * 3)
        
        # Generate target with some relationship to inputs
        output = input_feature * 0.5 + np.random.randn(n_samples) * 0.1
        
        return pd.DataFrame({
            'input_feature': input_feature,
            'constraint_feature': constraint_feature,
            'max_input': max_input,
            'output': output
        })
    
    def test_training_without_feature_tensors(self, sample_data):
        """Test standard training without feature tensors (baseline)."""
        config = {
            'input_features': [
                {'name': 'input_feature', 'type': 'number'}
            ],
            'output_features': [
                {
                    'name': 'output',
                    'type': 'number',
                    'loss': {'type': 'mean_squared_error'}  # Standard MSE loss
                }
            ],
            'trainer': {'epochs': 2, 'batch_size': 32}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LudwigModel(config, logging_level=40)
            train_stats, _, _ = model.train(dataset=sample_data)
            
            # Training should complete successfully
            assert 'training' in train_stats
            assert 'output' in train_stats['training']
            
            # Should have loss values
            assert 'loss' in train_stats['training']['output']
    
    def test_training_with_feature_tensors_disabled(self, sample_data):
        """Test training with physics loss but feature tensors disabled."""
        config = {
            'input_features': [
                {'name': 'input_feature', 'type': 'number'},
                {'name': 'constraint_feature', 'type': 'number'}
            ],
            'output_features': [
                {
                    'name': 'output',
                    'type': 'number',
                    'loss': {
                        'type': 'mean_squared_error',
                        'pass_input_features': False  # Disabled
                    }
                }
            ],
            'trainer': {'epochs': 2, 'batch_size': 32}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LudwigModel(config, logging_level=40)
            train_stats, _, _ = model.train(dataset=sample_data)
            
            # Should work like standard training
            assert 'training' in train_stats
            assert 'output' in train_stats['training']
    
    def test_training_with_feature_tensors_enabled(self, sample_data):
        """Test training with feature tensors enabled."""
        config = {
            'input_features': [
                {'name': 'input_feature', 'type': 'number'},
                {'name': 'constraint_feature', 'type': 'number'},
                {'name': 'max_input', 'type': 'number'}
            ],
            'output_features': [
                {
                    'name': 'output',
                    'type': 'number',
                    'loss': {
                        'type': 'mean_squared_error',
                        'pass_input_features': True  # Enable feature passing
                    }
                }
            ],
            'trainer': {'epochs': 2, 'batch_size': 32}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LudwigModel(config, logging_level=40)
            train_stats, _, _ = model.train(dataset=sample_data)
            
            # Should work with feature tensor support
            assert 'training' in train_stats
            assert 'output' in train_stats['training']
            
            # Loss values should be present
            assert 'loss' in train_stats['training']['output']
    
    def test_training_with_selected_features(self, sample_data):
        """Test training with specific feature selection."""
        config = {
            'input_features': [
                {'name': 'input_feature', 'type': 'number'},
                {'name': 'constraint_feature', 'type': 'number'},
                {'name': 'max_input', 'type': 'number'}
            ],
            'output_features': [
                {
                    'name': 'output',
                    'type': 'number',
                    'loss': {
                        'type': 'mean_squared_error',
                        'pass_input_features': True,
                        'input_feature_names': ['input_feature', 'constraint_feature']  # Only pass selected
                    }
                }
            ],
            'trainer': {'epochs': 2, 'batch_size': 32}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LudwigModel(config, logging_level=40)
            train_stats, _, _ = model.train(dataset=sample_data)
            
            # Should work with selected features
            assert 'training' in train_stats
            assert 'output' in train_stats['training']
    
    def test_multi_output_mixed_feature_passing(self, sample_data):
        """Test multi-output model with mixed feature passing settings."""
        # Add second output
        sample_data_multi = sample_data.copy()
        sample_data_multi['output2'] = sample_data['output'] * 1.5
        
        config = {
            'input_features': [
                {'name': 'input_feature', 'type': 'number'},
                {'name': 'constraint_feature', 'type': 'number'}
            ],
            'output_features': [
                {
                    'name': 'output',
                    'type': 'number',
                    'loss': {
                        'type': 'mean_squared_error',
                        'pass_input_features': True  # Use features
                    }
                },
                {
                    'name': 'output2',
                    'type': 'number',
                    'loss': {
                        'type': 'mean_squared_error',
                        'pass_input_features': False  # Don't use features
                    }
                }
            ],
            'trainer': {'epochs': 2, 'batch_size': 32}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LudwigModel(config, logging_level=40)
            train_stats, _, _ = model.train(dataset=sample_data_multi)
            
            # Both outputs should work
            assert 'training' in train_stats
            assert 'output' in train_stats['training']
            assert 'output2' in train_stats['training']
    
    def test_feature_tensor_memory_options(self, sample_data):
        """Test feature tensor memory optimization settings."""
        config = {
            'input_features': [
                {'name': 'input_feature', 'type': 'number'}
            ],
            'output_features': [
                {
                    'name': 'output',
                    'type': 'number',
                    'loss': {
                        'type': 'mean_squared_error',
                        'pass_input_features': True,
                        'detach_feature_tensors': True  # Memory optimization
                    }
                }
            ],
            'trainer': {'epochs': 2, 'batch_size': 32}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LudwigModel(config, logging_level=40)
            train_stats, _, _ = model.train(dataset=sample_data)
            
            # Should work with memory optimization
            assert 'training' in train_stats


class TestBackwardCompatibility:
    """Test backward compatibility with existing configurations."""
    
    @pytest.fixture
    def legacy_data(self):
        """Create data for legacy compatibility testing."""
        np.random.seed(42)
        n_samples = 100
        
        return pd.DataFrame({
            'feature': np.random.randn(n_samples),
            'target': np.random.randn(n_samples)
        })
    
    def test_legacy_config_unchanged(self, legacy_data):
        """Test that legacy configurations work unchanged."""
        # Old-style config with no feature tensor parameters
        config = {
            'input_features': [
                {'name': 'feature', 'type': 'number'}
            ],
            'output_features': [
                {
                    'name': 'target',
                    'type': 'number',
                    'loss': {'type': 'mean_squared_error'}  # No new parameters
                }
            ],
            'trainer': {'epochs': 1, 'batch_size': 16}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LudwigModel(config, logging_level=40)
            train_stats, _, _ = model.train(dataset=legacy_data)
            
            # Should work exactly as before
            assert 'training' in train_stats
            assert 'target' in train_stats['training']
    
    def test_gradual_migration(self, legacy_data):
        """Test gradual migration from legacy to feature tensor configs."""
        # Start with legacy config
        legacy_config = {
            'input_features': [{'name': 'feature', 'type': 'number'}],
            'output_features': [{'name': 'target', 'type': 'number', 'loss': {'type': 'mean_squared_error'}}],
            'trainer': {'epochs': 1, 'batch_size': 16}
        }
        
        # Migrate to feature tensor config
        modern_config = {
            'input_features': [{'name': 'feature', 'type': 'number'}],
            'output_features': [{
                'name': 'target', 
                'type': 'number', 
                'loss': {
                    'type': 'mean_squared_error',
                    'pass_input_features': True  # Add new capability
                }
            }],
            'trainer': {'epochs': 1, 'batch_size': 16}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Both should work
            legacy_model = LudwigModel(legacy_config, logging_level=40)
            legacy_stats, _, _ = legacy_model.train(dataset=legacy_data)
            
            modern_model = LudwigModel(modern_config, logging_level=40)
            modern_stats, _, _ = modern_model.train(dataset=legacy_data)
            
            # Both should complete successfully
            assert 'training' in legacy_stats
            assert 'training' in modern_stats


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def error_test_data(self):
        """Create data for error testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'target': np.random.randn(50)
        })
    
    def test_missing_input_features_graceful_handling(self, error_test_data):
        """Test graceful handling when specified input features are missing."""
        config = {
            'input_features': [
                {'name': 'feature1', 'type': 'number'}
                # feature2 not included as input
            ],
            'output_features': [
                {
                    'name': 'target',
                    'type': 'number',
                    'loss': {
                        'type': 'mean_squared_error',
                        'pass_input_features': True,
                        'input_feature_names': ['feature1', 'missing_feature']  # One missing
                    }
                }
            ],
            'trainer': {'epochs': 1, 'batch_size': 16}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model = LudwigModel(config, logging_level=40)
            # Should not crash, should handle gracefully
            train_stats, _, _ = model.train(dataset=error_test_data[['feature1', 'target']])
            
            assert 'training' in train_stats
    
    def test_no_input_features_with_feature_tensors_enabled(self, error_test_data):
        """Test behavior when feature tensors are enabled but no input features exist."""
        config = {
            'input_features': [],  # No input features
            'output_features': [
                {
                    'name': 'target',
                    'type': 'number',
                    'loss': {
                        'type': 'mean_squared_error',
                        'pass_input_features': True  # Enabled but no features
                    }
                }
            ],
            'trainer': {'epochs': 1, 'batch_size': 16}
        }
        
        # This should either work gracefully or raise a clear error
        # (depending on Ludwig's validation logic)
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises((ValueError, RuntimeError)):
                model = LudwigModel(config, logging_level=40)
                model.train(dataset=error_test_data[['target']])


class TestPerformanceBasics:
    """Basic performance and overhead tests."""
    
    @pytest.fixture
    def performance_data(self):
        """Create larger dataset for basic performance testing."""
        np.random.seed(42)
        n_samples = 1000  # Larger but still reasonable for tests
        
        return pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples),
            'target': np.random.randn(n_samples)
        })
    
    def test_training_speed_baseline_vs_features(self, performance_data):
        """Compare training speed with and without feature tensor passing."""
        import time
        
        # Baseline config (no feature tensors)
        baseline_config = {
            'input_features': [
                {'name': 'feature1', 'type': 'number'},
                {'name': 'feature2', 'type': 'number'},
                {'name': 'feature3', 'type': 'number'}
            ],
            'output_features': [
                {
                    'name': 'target',
                    'type': 'number',
                    'loss': {'type': 'mean_squared_error'}
                }
            ],
            'trainer': {'epochs': 3, 'batch_size': 64}
        }
        
        # Feature tensor config
        feature_config = baseline_config.copy()
        feature_config['output_features'][0]['loss']['pass_input_features'] = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Baseline timing
            start_time = time.time()
            baseline_model = LudwigModel(baseline_config, logging_level=40)
            baseline_model.train(dataset=performance_data)
            baseline_time = time.time() - start_time
            
            # Feature tensor timing
            start_time = time.time()
            feature_model = LudwigModel(feature_config, logging_level=40)
            feature_model.train(dataset=performance_data)
            feature_time = time.time() - start_time
            
            # Calculate overhead
            overhead_pct = ((feature_time - baseline_time) / baseline_time) * 100
            
            # For standard losses that ignore features, overhead should be minimal
            # This is a basic check - detailed benchmarks would be in separate scripts
            print(f"Training time overhead: {overhead_pct:.2f}%")
            
            # Both should complete successfully
            assert baseline_time > 0
            assert feature_time > 0