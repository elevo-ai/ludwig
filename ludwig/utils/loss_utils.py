import logging
from typing import Dict, List, Optional, Set

import torch

# Type alias for feature tensor dictionary
FeatureTensorDict = Dict[str, torch.Tensor]

logger = logging.getLogger(__name__)


def rmspe_loss(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    """Root mean square percentage error.

    Bad predictions can lead to arbitrarily large RMSPE values, especially if some values of targets are very close to
    zero. We return a large value instead of inf when (some) targets are zero.
    """
    epsilon = 1e-4
    # add epsilon if targets are zero to avoid division by zero
    denominator = targets + epsilon * (targets == 0).float()
    loss = torch.sqrt(torch.mean(((targets - predictions).float() / denominator) ** 2))
    return loss


def mean_confidence_penalty(probabilities: torch.Tensor, num_classes: int) -> torch.Tensor:
    max_entropy = torch.log(torch.tensor(num_classes))
    # clipping needed for avoiding log(0) = -inf
    entropy_per_class, _ = torch.max(-probabilities * torch.log(torch.clamp(probabilities, 1e-10, 1)), dim=0)
    entropy = torch.sum(entropy_per_class, -1)
    penalty = (max_entropy - entropy) / max_entropy
    return torch.mean(penalty)


# ============================================================================
# Feature Tensor Extraction Utilities for Physics-Informed Losses
# ============================================================================

def extract_feature_tensors(
    batch: Dict[str, torch.Tensor],
    input_feature_names: List[str],
    selected_features: Optional[List[str]] = None,
    detach: bool = True
) -> FeatureTensorDict:
    """
    Extract input feature tensors from batch for passing to loss functions.
    
    Args:
        batch: Full batch dictionary containing all features and targets
        input_feature_names: List of all available input feature names
        selected_features: Subset of features to extract (None = all features)
        detach: Whether to detach tensors from computation graph (default True)
    
    Returns:
        Dictionary mapping feature names to tensors
        
    Raises:
        ValueError: If selected features don't exist in input features
        KeyError: If required features are missing from batch
    """
    # Determine which features to extract
    if selected_features is None:
        features_to_extract = input_feature_names
    else:
        # Validate selected features exist in input features
        invalid_features = set(selected_features) - set(input_feature_names)
        if invalid_features:
            raise ValueError(
                f"Selected features {invalid_features} not found in input features. "
                f"Available input features: {input_feature_names}"
            )
        features_to_extract = selected_features
    
    # Extract tensors
    feature_tensors = {}
    missing_features = []
    
    for feature_name in features_to_extract:
        if feature_name in batch:
            tensor = batch[feature_name]
            
            # Ensure it's a tensor
            if not isinstance(tensor, torch.Tensor):
                try:
                    tensor = torch.tensor(tensor)
                except (TypeError, ValueError) as e:
                    logger.warning(
                        f"Could not convert feature '{feature_name}' to tensor: {e}"
                    )
                    continue
            
            # Detach if requested (default behavior for memory efficiency)
            if detach:
                tensor = tensor.detach()
            
            feature_tensors[feature_name] = tensor
        else:
            missing_features.append(feature_name)
    
    # Log missing features (may be expected in some cases)
    if missing_features:
        logger.debug(
            f"Features not found in batch: {missing_features}. "
            f"Available keys: {list(batch.keys())}"
        )
    
    return feature_tensors


def validate_feature_tensors(
    feature_tensors: FeatureTensorDict,
    expected_batch_size: int,
    feature_names: Optional[Set[str]] = None
) -> None:
    """
    Validate feature tensor dictionary for consistency and correctness.
    
    Args:
        feature_tensors: Dictionary of feature tensors to validate
        expected_batch_size: Expected batch size for all tensors
        feature_names: Optional set of expected feature names
    
    Raises:
        ValueError: If validation fails
        TypeError: If tensors have wrong type
    """
    if not isinstance(feature_tensors, dict):
        raise TypeError(f"feature_tensors must be dict, got {type(feature_tensors)}")
    
    if len(feature_tensors) == 0:
        logger.warning("feature_tensors dictionary is empty")
        return
    
    # Validate each tensor
    for feature_name, tensor in feature_tensors.items():
        # Check tensor type
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Feature '{feature_name}' must be torch.Tensor, got {type(tensor)}"
            )
        
        # Check batch size consistency
        if len(tensor.shape) == 0:  # Scalar tensor
            raise ValueError(
                f"Feature '{feature_name}' is a scalar tensor, expected batch dimension"
            )
        
        if tensor.shape[0] != expected_batch_size:
            raise ValueError(
                f"Feature '{feature_name}' has batch size {tensor.shape[0]}, "
                f"expected {expected_batch_size}"
            )
        
        # Check for NaN or Inf values
        if torch.isnan(tensor).any():
            logger.warning(f"Feature '{feature_name}' contains NaN values")
        
        if torch.isinf(tensor).any():
            logger.warning(f"Feature '{feature_name}' contains infinite values")
    
    # Validate expected feature names if provided
    if feature_names is not None:
        missing_features = feature_names - set(feature_tensors.keys())
        if missing_features:
            raise ValueError(
                f"Expected features {missing_features} not found in feature_tensors"
            )
        
        extra_features = set(feature_tensors.keys()) - feature_names
        if extra_features:
            logger.debug(f"Extra features found: {extra_features}")


def check_feature_tensor_compatibility(
    loss_function: torch.nn.Module,
    requires_features: bool = False,
    required_feature_names: Optional[List[str]] = None
) -> bool:
    """
    Check if a loss function is compatible with feature tensor passing.
    
    Args:
        loss_function: The loss function module to check
        requires_features: Whether this loss function requires features
        required_feature_names: List of feature names required by the loss
    
    Returns:
        True if compatible, False otherwise
    """
    # Check if loss function has the expected forward signature
    forward_method = getattr(loss_function, 'forward', None)
    if forward_method is None:
        return False
    
    try:
        import inspect
        sig = inspect.signature(forward_method)
        params = list(sig.parameters.keys())
        
        # Expected signature: forward(self, predictions, targets, feature_tensors=None)
        # or: forward(predictions, targets, feature_tensors=None) for functions
        expected_params = ['predictions', 'targets', 'feature_tensors']
        
        # Check if all expected parameters are present (excluding 'self')
        if 'self' in params:
            params = params[1:]  # Remove 'self'
        
        if len(params) < 2:
            return False
        
        # At minimum, should have predictions and targets
        if params[0] not in ['predictions', 'preds'] or params[1] not in ['targets', 'target']:
            return False
        
        # Check for feature_tensors parameter
        has_feature_tensors_param = 'feature_tensors' in params
        
        if requires_features and not has_feature_tensors_param:
            logger.warning(
                f"Loss function {loss_function.__class__.__name__} requires features "
                f"but doesn't have feature_tensors parameter"
            )
            return False
        
        return True
        
    except Exception as e:
        logger.debug(f"Could not inspect loss function signature: {e}")
        return True  # Assume compatible if we can't check


def log_feature_tensor_info(
    feature_tensors: FeatureTensorDict,
    prefix: str = "Feature tensors"
) -> None:
    """
    Log information about feature tensors for debugging.
    
    Args:
        feature_tensors: Dictionary of feature tensors
        prefix: Prefix for log messages
    """
    if not feature_tensors:
        logger.debug(f"{prefix}: empty")
        return
    
    logger.debug(f"{prefix}: {len(feature_tensors)} features")
    
    for name, tensor in feature_tensors.items():
        logger.debug(
            f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
            f"device={tensor.device}, requires_grad={tensor.requires_grad}"
        )


def estimate_feature_tensor_memory_overhead(
    feature_tensors: FeatureTensorDict,
    base_batch_memory: Optional[float] = None
) -> Dict[str, float]:
    """
    Estimate memory overhead of feature tensors.
    
    Args:
        feature_tensors: Dictionary of feature tensors
        base_batch_memory: Base memory usage without features (in MB)
    
    Returns:
        Dictionary with memory information
    """
    total_elements = 0
    total_bytes = 0
    
    dtype_bytes = {
        torch.float32: 4,
        torch.float64: 8,
        torch.float16: 2,
        torch.int32: 4,
        torch.int64: 8,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
    }
    
    for name, tensor in feature_tensors.items():
        elements = tensor.numel()
        bytes_per_element = dtype_bytes.get(tensor.dtype, 4)  # Default to 4 bytes
        tensor_bytes = elements * bytes_per_element
        
        total_elements += elements
        total_bytes += tensor_bytes
    
    total_mb = total_bytes / (1024 * 1024)
    
    info = {
        "total_elements": total_elements,
        "total_bytes": total_bytes,
        "total_mb": total_mb,
        "num_tensors": len(feature_tensors)
    }
    
    if base_batch_memory is not None:
        info["overhead_pct"] = (total_mb / base_batch_memory) * 100 if base_batch_memory > 0 else 0
    
    return info
