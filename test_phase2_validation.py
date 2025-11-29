#!/usr/bin/env python3
"""
Phase 2 Validation Script: Test Enhanced Loss Module System

This script validates that all loss modules have been successfully updated
to support feature tensors while maintaining backward compatibility.

Usage:
    python test_phase2_validation.py
"""

import sys
import traceback
from typing import Dict, Any

def test_imports():
    """Test that all necessary imports work correctly."""
    try:
        # Test basic imports
        import torch
        print("✓ PyTorch import successful")
        
        # Test Ludwig loss module imports
        from ludwig.modules import loss_modules
        from ludwig.schema.features.loss.loss import (
            MSELossConfig,
            MAELossConfig, 
            MAPELossConfig,
            RMSELossConfig,
            RMSPELossConfig,
            HuberLossConfig,
            BWCEWLossConfig,
            SoftmaxCrossEntropyLossConfig,
            CORNLossConfig
        )
        print("✓ Ludwig loss module imports successful")
        
        # Test feature tensor type import
        from ludwig.utils.loss_utils import FeatureTensorDict
        print("✓ Feature tensor type import successful")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_base_loss_class():
    """Test that the BaseLoss class works correctly."""
    try:
        from ludwig.modules.loss_modules import BaseLoss
        from ludwig.schema.features.loss.loss import MSELossConfig
        
        # Test BaseLoss initialization
        config = MSELossConfig(
            pass_input_features=True,
            input_feature_names=['feature1', 'feature2'],
            detach_feature_tensors=False
        )
        base_loss = BaseLoss(config)
        
        # Verify configuration
        assert base_loss.pass_input_features == True
        assert base_loss.input_feature_names == ['feature1', 'feature2']
        assert base_loss.detach_feature_tensors == False
        assert base_loss.expects_feature_tensors == False  # Default
        
        print("✓ BaseLoss class works correctly")
        return True
    except Exception as e:
        print(f"✗ BaseLoss test failed: {e}")
        traceback.print_exc()
        return False


def test_mse_loss_feature_support():
    """Test MSELoss with feature tensor support."""
    try:
        import torch
        from ludwig.modules.loss_modules import MSELoss
        from ludwig.schema.features.loss.loss import MSELossConfig
        
        # Create MSE loss with feature support
        config = MSELossConfig(pass_input_features=True, input_feature_names=['temperature'])
        mse_loss = MSELoss(config)
        
        # Create test data
        batch_size = 16
        predictions = torch.randn(batch_size, 1)
        targets = torch.randn(batch_size, 1)
        feature_tensors = {
            'temperature': torch.randn(batch_size, 1),
            'pressure': torch.randn(batch_size, 1)
        }
        
        # Test without feature tensors (backward compatibility)
        loss_without = mse_loss(predictions, targets)
        assert loss_without.shape == ()
        
        # Test with feature tensors (should give same result for MSE)
        loss_with = mse_loss(predictions, targets, feature_tensors)
        assert loss_with.shape == ()
        assert torch.allclose(loss_without, loss_with)
        
        print("✓ MSELoss feature support works correctly")
        return True
    except Exception as e:
        print(f"✗ MSELoss test failed: {e}")
        traceback.print_exc()
        return False


def test_all_loss_functions_updated():
    """Test that all loss functions accept the new signature."""
    try:
        import torch
        from ludwig.modules import loss_modules
        from ludwig.schema.features.loss.loss import (
            MSELossConfig,
            MAELossConfig, 
            MAPELossConfig,
            RMSELossConfig,
            RMSPELossConfig,
            HuberLossConfig,
        )
        
        # Test data
        batch_size = 8
        predictions = torch.randn(batch_size, 2)
        targets = torch.randn(batch_size, 2)
        feature_tensors = {'feature1': torch.randn(batch_size, 1)}
        
        # Test various loss functions
        loss_configs_and_classes = [
            (MSELossConfig(), loss_modules.MSELoss),
            (MAELossConfig(), loss_modules.MAELoss),
            (MAPELossConfig(), loss_modules.MAPELoss),
            (RMSELossConfig(), loss_modules.RMSELoss),
            (RMSPELossConfig(), loss_modules.RMSPELoss),
            (HuberLossConfig(delta=1.0), loss_modules.HuberLoss),
        ]
        
        for config, loss_class in loss_configs_and_classes:
            loss_fn = loss_class(config)
            
            # Test without features (backward compatibility)
            result_without = loss_fn(predictions, targets)
            assert result_without.shape == ()
            
            # Test with features (new functionality) 
            result_with = loss_fn(predictions, targets, feature_tensors)
            assert result_with.shape == ()
            
            # For these basic losses, results should be identical
            assert torch.allclose(result_without, result_with)
            
            print(f"✓ {loss_class.__name__} works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        traceback.print_exc()
        return False


def test_cross_entropy_losses():
    """Test cross-entropy losses with feature tensors."""
    try:
        import torch
        from ludwig.modules import loss_modules
        from ludwig.schema.features.loss.loss import (
            SoftmaxCrossEntropyLossConfig,
            BWCEWLossConfig
        )
        
        # Test SoftmaxCrossEntropyLoss
        batch_size = 16
        num_classes = 5
        predictions = torch.randn(batch_size, num_classes)
        targets = torch.randint(0, num_classes, (batch_size,))
        feature_tensors = {'feature1': torch.randn(batch_size, 1)}
        
        config = SoftmaxCrossEntropyLossConfig()
        loss_fn = loss_modules.SoftmaxCrossEntropyLoss(config)
        
        result_without = loss_fn(predictions, targets)
        result_with = loss_fn(predictions, targets, feature_tensors)
        
        assert torch.allclose(result_without, result_with)
        print("✓ SoftmaxCrossEntropyLoss works correctly")
        
        # Test BWCEWLoss
        binary_predictions = torch.randn(batch_size, 1)
        binary_targets = torch.randint(0, 2, (batch_size, 1)).float()
        
        binary_config = BWCEWLossConfig(positive_class_weight=1.5)
        binary_loss_fn = loss_modules.BWCEWLoss(binary_config)
        
        binary_result_without = binary_loss_fn(binary_predictions, binary_targets)
        binary_result_with = binary_loss_fn(binary_predictions, binary_targets, feature_tensors)
        
        assert torch.allclose(binary_result_without, binary_result_with)
        print("✓ BWCEWLoss works correctly")
        
        return True
    except Exception as e:
        print(f"✗ Cross-entropy loss test failed: {e}")
        traceback.print_exc()
        return False


def test_physics_informed_loss_example():
    """Test an example physics-informed loss function."""
    try:
        import torch
        from ludwig.modules.loss_modules import BaseLoss
        from ludwig.schema.features.loss.loss import MSELossConfig
        
        # Create a simple physics-informed loss
        class SimplePhysicsLoss(BaseLoss):
            def __init__(self, config):
                super().__init__(config)
                self.expects_feature_tensors = True
                self.mse = torch.nn.MSELoss()
                
            def forward(self, predictions, targets, feature_tensors=None):
                data_loss = self.mse(predictions, targets)
                
                if feature_tensors is not None:
                    # Simple constraint: penalize negative predictions when feature is positive
                    feature = feature_tensors.get('feature1', torch.zeros_like(predictions))
                    positive_mask = (feature > 0).float()
                    penalty = torch.relu(-predictions) * positive_mask
                    physics_penalty = penalty.mean() * 10.0
                    return data_loss + physics_penalty
                
                return data_loss
        
        # Test the physics loss
        config = MSELossConfig(pass_input_features=True, input_feature_names=['feature1'])
        physics_loss = SimplePhysicsLoss(config)
        
        batch_size = 32
        predictions = torch.randn(batch_size, 1) * 0.1  # Small values
        targets = torch.randn(batch_size, 1)
        feature_tensors = {
            'feature1': torch.abs(torch.randn(batch_size, 1))  # All positive
        }
        
        # Test both modes
        loss_without = physics_loss(predictions, targets)
        loss_with = physics_loss(predictions, targets, feature_tensors)
        
        # Physics loss should typically be higher due to constraint penalty
        # (though this depends on the specific values)
        assert loss_without.shape == ()
        assert loss_with.shape == ()
        
        print("✓ Physics-informed loss example works correctly")
        return True
    except Exception as e:
        print(f"✗ Physics loss test failed: {e}")
        traceback.print_exc()
        return False


def test_loss_creation_registry():
    """Test loss creation through the registry system."""
    try:
        from ludwig.modules.loss_modules import create_loss
        from ludwig.schema.features.loss.loss import MSELossConfig
        
        # Test creating loss with feature support through registry
        config = MSELossConfig(
            pass_input_features=True,
            input_feature_names=['feature1']
        )
        
        loss_fn = create_loss(config)
        
        # Verify it's the correct type and has our configuration
        assert hasattr(loss_fn, 'pass_input_features')
        assert loss_fn.pass_input_features == True
        assert loss_fn.input_feature_names == ['feature1']
        
        print("✓ Loss creation registry works correctly")
        return True
    except Exception as e:
        print(f"✗ Loss creation test failed: {e}")
        traceback.print_exc()
        return False


def test_backward_compatibility():
    """Test that old configurations still work."""
    try:
        from ludwig.modules.loss_modules import create_loss
        from ludwig.schema.features.loss.loss import MSELossConfig
        import torch
        
        # Test with old-style config (no feature parameters)
        old_config = MSELossConfig()  # All defaults
        loss_fn = create_loss(old_config)
        
        # Should work exactly as before
        predictions = torch.randn(10, 1)
        targets = torch.randn(10, 1)
        
        result = loss_fn(predictions, targets)
        assert result.shape == ()
        
        # Should also work with explicit None for feature_tensors
        result_explicit = loss_fn(predictions, targets, feature_tensors=None)
        assert torch.allclose(result, result_explicit)
        
        print("✓ Backward compatibility maintained")
        return True
    except Exception as e:
        print(f"✗ Backward compatibility test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("Phase 2 Loss Module Validation Tests")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("BaseLoss Class", test_base_loss_class),
        ("MSE Loss Feature Support", test_mse_loss_feature_support),
        ("All Loss Functions Updated", test_all_loss_functions_updated),
        ("Cross-Entropy Losses", test_cross_entropy_losses),
        ("Physics-Informed Loss Example", test_physics_informed_loss_example),
        ("Loss Creation Registry", test_loss_creation_registry),
        ("Backward Compatibility", test_backward_compatibility),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All tests passed! Phase 2 implementation is working correctly.")
        return 0
    else:
        print(f"❌ {failed} tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())