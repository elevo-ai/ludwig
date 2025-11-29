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
"""Tests for feature tensor extraction utilities for physics-informed losses."""

import pytest
import torch

from ludwig.utils.loss_utils import (
    FeatureTensorDict,
    check_feature_tensor_compatibility,
    estimate_feature_tensor_memory_overhead,
    extract_feature_tensors,
    log_feature_tensor_info,
    validate_feature_tensors,
)


class TestFeatureExtraction:
    """Test feature tensor extraction functionality."""
    
    def test_extract_all_features(self):
        """Test extracting all input features."""
        batch = {
            'feature1': torch.randn(32, 1),
            'feature2': torch.randn(32, 2),
            'feature3': torch.randn(32, 3),
            'target': torch.randn(32, 1)
        }
        input_features = ['feature1', 'feature2', 'feature3']
        
        result = extract_feature_tensors(batch, input_features)
        
        assert 'feature1' in result
        assert 'feature2' in result
        assert 'feature3' in result
        assert 'target' not in result  # Should not extract targets
        assert result['feature1'].shape == (32, 1)
        assert result['feature2'].shape == (32, 2)
        assert result['feature3'].shape == (32, 3)
    
    def test_extract_selected_features(self):
        """Test extracting only selected features."""
        batch = {
            'feature1': torch.randn(32, 1),
            'feature2': torch.randn(32, 2),
            'feature3': torch.randn(32, 3)
        }
        input_features = ['feature1', 'feature2', 'feature3']
        selected_features = ['feature1', 'feature3']
        
        result = extract_feature_tensors(
            batch, 
            input_features,
            selected_features=selected_features
        )
        
        assert 'feature1' in result
        assert 'feature3' in result
        assert 'feature2' not in result
        assert len(result) == 2
    
    def test_detach_by_default(self):
        """Test that features are detached by default."""
        batch = {
            'feature1': torch.randn(32, 1, requires_grad=True)
        }
        input_features = ['feature1']
        
        result = extract_feature_tensors(batch, input_features, detach=True)
        
        assert not result['feature1'].requires_grad
    
    def test_no_detach_when_disabled(self):
        """Test keeping gradients when detach is disabled."""
        batch = {
            'feature1': torch.randn(32, 1, requires_grad=True)
        }
        input_features = ['feature1']
        
        result = extract_feature_tensors(batch, input_features, detach=False)
        
        assert result['feature1'].requires_grad
    
    def test_invalid_selected_features(self):
        """Test error handling for invalid selected features."""
        batch = {
            'feature1': torch.randn(32, 1),
            'feature2': torch.randn(32, 2)
        }
        input_features = ['feature1', 'feature2']
        selected_features = ['feature1', 'invalid_feature']
        
        with pytest.raises(ValueError, match="not found in input features"):
            extract_feature_tensors(batch, input_features, selected_features=selected_features)
    
    def test_missing_features_in_batch(self):
        """Test handling of features missing from batch."""
        batch = {
            'feature1': torch.randn(32, 1)
            # feature2 is missing
        }
        input_features = ['feature1', 'feature2']
        
        # Should not raise error, just log debug message
        result = extract_feature_tensors(batch, input_features)
        
        assert 'feature1' in result
        assert 'feature2' not in result
        assert len(result) == 1
    
    def test_convert_non_tensor_to_tensor(self):
        """Test conversion of non-tensor values to tensors."""
        import numpy as np
        
        batch = {
            'feature1': torch.randn(32, 1),
            'feature2': np.random.randn(32, 2)  # NumPy array
        }
        input_features = ['feature1', 'feature2']
        
        result = extract_feature_tensors(batch, input_features)
        
        assert isinstance(result['feature1'], torch.Tensor)
        assert isinstance(result['feature2'], torch.Tensor)
        assert result['feature2'].shape == (32, 2)


class TestFeatureTensorValidation:
    """Test feature tensor validation functionality."""
    
    def test_valid_feature_tensors(self):
        """Test validation of valid feature tensors."""
        feature_tensors = {
            'feature1': torch.randn(32, 1),
            'feature2': torch.randn(32, 2)
        }
        
        # Should not raise any errors
        validate_feature_tensors(feature_tensors, expected_batch_size=32)
    
    def test_batch_size_mismatch(self):
        """Test validation catches batch size mismatches."""
        feature_tensors = {
            'feature1': torch.randn(32, 1),
            'feature2': torch.randn(16, 2)  # Wrong batch size
        }
        
        with pytest.raises(ValueError, match="batch size"):
            validate_feature_tensors(feature_tensors, expected_batch_size=32)
    
    def test_scalar_tensor_error(self):
        """Test validation catches scalar tensors."""
        feature_tensors = {
            'feature1': torch.tensor(5.0)  # Scalar tensor
        }
        
        with pytest.raises(ValueError, match="scalar tensor"):
            validate_feature_tensors(feature_tensors, expected_batch_size=32)
    
    def test_non_tensor_values(self):
        """Test validation catches non-tensor values."""
        feature_tensors = {
            'feature1': [1, 2, 3]  # List instead of tensor
        }
        
        with pytest.raises(TypeError, match="must be torch.Tensor"):
            validate_feature_tensors(feature_tensors, expected_batch_size=3)
    
    def test_non_dict_input(self):
        """Test validation catches non-dict input."""
        with pytest.raises(TypeError, match="must be dict"):
            validate_feature_tensors([1, 2, 3], expected_batch_size=3)
    
    def test_expected_feature_names_validation(self):
        """Test validation of expected feature names."""
        feature_tensors = {
            'feature1': torch.randn(32, 1)
            # feature2 is missing
        }
        expected_features = {'feature1', 'feature2'}
        
        with pytest.raises(ValueError, match="Expected features"):
            validate_feature_tensors(
                feature_tensors, 
                expected_batch_size=32,
                feature_names=expected_features
            )
    
    def test_empty_feature_tensors(self):
        """Test handling of empty feature tensor dict."""
        feature_tensors = {}
        
        # Should log warning but not raise error
        validate_feature_tensors(feature_tensors, expected_batch_size=32)
    
    def test_nan_and_inf_detection(self, caplog):
        """Test detection of NaN and infinite values."""
        feature_tensors = {
            'feature_with_nan': torch.tensor([[float('nan'), 2.0], [3.0, 4.0]]),
            'feature_with_inf': torch.tensor([[float('inf'), 2.0], [3.0, 4.0]])
        }
        
        with caplog.at_level('WARNING'):
            validate_feature_tensors(feature_tensors, expected_batch_size=2)
        
        assert "contains NaN values" in caplog.text
        assert "contains infinite values" in caplog.text


class TestCompatibilityChecking:
    """Test loss function compatibility checking."""
    
    def test_compatible_loss_function(self):
        """Test detection of compatible loss function."""
        class CompatibleLoss(torch.nn.Module):
            def forward(self, predictions, targets, feature_tensors=None):
                return torch.tensor(0.0)
        
        loss = CompatibleLoss()
        assert check_feature_tensor_compatibility(loss)
    
    def test_old_style_loss_function(self):
        """Test detection of old-style loss function."""
        class OldStyleLoss(torch.nn.Module):
            def forward(self, predictions, targets):
                return torch.tensor(0.0)
        
        loss = OldStyleLoss()
        # Should still be compatible (backward compatibility)
        assert check_feature_tensor_compatibility(loss)
    
    def test_requires_features_but_incompatible(self):
        """Test detection of loss that requires features but is incompatible."""
        class IncompatibleLoss(torch.nn.Module):
            def forward(self, predictions, targets):
                return torch.tensor(0.0)
        
        loss = IncompatibleLoss()
        # Should return False when requires_features=True but no feature_tensors param
        assert not check_feature_tensor_compatibility(loss, requires_features=True)
    
    def test_no_forward_method(self):
        """Test handling of objects without forward method."""
        class NoForwardMethod:
            pass
        
        obj = NoForwardMethod()
        assert not check_feature_tensor_compatibility(obj)


class TestUtilityFunctions:
    """Test utility functions for debugging and monitoring."""
    
    def test_log_feature_tensor_info(self, caplog):
        """Test logging of feature tensor information."""
        feature_tensors = {
            'feature1': torch.randn(32, 1),
            'feature2': torch.randn(32, 2, dtype=torch.float16)
        }
        
        with caplog.at_level('DEBUG'):
            log_feature_tensor_info(feature_tensors, prefix="Test features")
        
        assert "Test features: 2 features" in caplog.text
        assert "feature1: shape=torch.Size([32, 1])" in caplog.text
        assert "feature2: shape=torch.Size([32, 2])" in caplog.text
    
    def test_log_empty_feature_tensors(self, caplog):
        """Test logging of empty feature tensor dict."""
        feature_tensors = {}
        
        with caplog.at_level('DEBUG'):
            log_feature_tensor_info(feature_tensors)
        
        assert "Feature tensors: empty" in caplog.text
    
    def test_estimate_memory_overhead(self):
        """Test memory overhead estimation."""
        feature_tensors = {
            'feature1': torch.randn(32, 1, dtype=torch.float32),  # 32 * 1 * 4 = 128 bytes
            'feature2': torch.randn(32, 2, dtype=torch.float32)   # 32 * 2 * 4 = 256 bytes
        }
        
        info = estimate_feature_tensor_memory_overhead(feature_tensors)
        
        assert info['total_elements'] == 32 + 64  # 32*1 + 32*2
        assert info['total_bytes'] == 128 + 256    # 384 bytes total
        assert info['total_mb'] == (128 + 256) / (1024 * 1024)
        assert info['num_tensors'] == 2
    
    def test_estimate_memory_overhead_with_baseline(self):
        """Test memory overhead estimation with baseline."""
        feature_tensors = {
            'feature1': torch.randn(32, 1, dtype=torch.float32)  # 128 bytes
        }
        base_memory_mb = 1.0  # 1 MB baseline
        
        info = estimate_feature_tensor_memory_overhead(
            feature_tensors, 
            base_batch_memory=base_memory_mb
        )
        
        expected_overhead_pct = (128 / (1024 * 1024)) / base_memory_mb * 100
        assert abs(info['overhead_pct'] - expected_overhead_pct) < 0.01
    
    def test_different_tensor_dtypes(self):
        """Test memory estimation with different tensor dtypes."""
        feature_tensors = {
            'float32_feature': torch.randn(10, 1, dtype=torch.float32),  # 10 * 4 = 40 bytes
            'float16_feature': torch.randn(10, 1, dtype=torch.float16),  # 10 * 2 = 20 bytes
            'int64_feature': torch.randint(0, 10, (10, 1), dtype=torch.int64)  # 10 * 8 = 80 bytes
        }
        
        info = estimate_feature_tensor_memory_overhead(feature_tensors)
        
        assert info['total_bytes'] == 40 + 20 + 80  # 140 bytes total
        assert info['num_tensors'] == 3


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_batch_size(self):
        """Test handling of zero batch size."""
        feature_tensors = {
            'feature1': torch.empty(0, 1)  # Empty tensor
        }
        
        validate_feature_tensors(feature_tensors, expected_batch_size=0)
    
    def test_very_large_tensors(self):
        """Test handling of very large tensors."""
        # Create a reasonably large tensor for testing
        large_tensor = torch.randn(1000, 100)  # Should not cause memory issues in test
        feature_tensors = {'large_feature': large_tensor}
        
        validate_feature_tensors(feature_tensors, expected_batch_size=1000)
        
        info = estimate_feature_tensor_memory_overhead(feature_tensors)
        assert info['total_elements'] == 100000
    
    def test_mixed_devices(self):
        """Test handling of tensors on different devices."""
        feature_tensors = {
            'cpu_feature': torch.randn(32, 1, device='cpu')
        }
        
        # Add GPU tensor only if CUDA is available
        if torch.cuda.is_available():
            feature_tensors['gpu_feature'] = torch.randn(32, 1, device='cuda')
        
        # Should still validate successfully
        validate_feature_tensors(feature_tensors, expected_batch_size=32)
    
    def test_unsupported_dtype_fallback(self):
        """Test fallback behavior for unsupported dtypes."""
        # Use a dtype not in the mapping
        feature_tensors = {
            'bool_feature': torch.randint(0, 2, (32, 1), dtype=torch.bool)
        }
        
        info = estimate_feature_tensor_memory_overhead(feature_tensors)
        
        # Should fall back to 4 bytes per element
        assert info['total_bytes'] == 32 * 4  # 32 elements * 4 bytes (fallback)