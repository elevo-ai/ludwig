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


from typing import Optional, Type

import torch
from torch import nn, Tensor
from torch.nn import HuberLoss as _HuberLoss
from torch.nn import L1Loss
from torch.nn import MSELoss as _MSELoss
from torchmetrics.functional import mean_absolute_percentage_error

import ludwig.utils.loss_utils as utils
from ludwig.constants import LOGITS
from ludwig.modules.loss_implementations.corn import corn_loss

# Feature tensor support
FeatureTensorDict = utils.FeatureTensorDict
from ludwig.schema.features.loss.loss import (
    BaseLossConfig,
    BWCEWLossConfig,
    CORNLossConfig,
    HuberLossConfig,
    MAELossConfig,
    MAPELossConfig,
    MSELossConfig,
    NextTokenSoftmaxCrossEntropyLossConfig,
    RMSELossConfig,
    RMSPELossConfig,
    SequenceSoftmaxCrossEntropyLossConfig,
    SigmoidCrossEntropyLossConfig,
    SoftmaxCrossEntropyLossConfig,
)
from ludwig.utils import strings_utils
from ludwig.utils.registry import Registry

# used for Laplace smoothing for candidate samplers
EPSILON = 1.0e-10

loss_impl_registry = Registry[Type[nn.Module]]()


def register_loss(config_cls: Type[BaseLossConfig]):
    def wrap(cls: Type[nn.Module]):
        loss_impl_registry[config_cls] = cls
        return cls

    return wrap


def create_loss(config: BaseLossConfig) -> nn.Module:
    return loss_impl_registry[type(config)](config)


class BaseLoss(nn.Module):
    """
    Enhanced base class for all Ludwig loss functions with feature tensor support.
    
    This class provides the foundation for physics-informed losses that can access
    input features during training. All loss functions should inherit from this
    class to maintain consistency and enable feature tensor support.
    
    Subclasses should implement forward() with the signature:
        forward(predictions, targets, feature_tensors=None)
    
    The feature_tensors parameter is optional and will only be provided if
    pass_input_features=true in the loss configuration.
    """
    
    def __init__(self, config: BaseLossConfig = None):
        super().__init__()
        # Store configuration for feature tensor handling
        self.config = config
        
        # Flag indicating if this loss expects/uses feature tensors
        # Can be overridden by subclasses that require features
        self.expects_feature_tensors = False
        
        # Store feature tensor configuration if available
        if config:
            self.pass_input_features = getattr(config, 'pass_input_features', False)
            self.input_feature_names = getattr(config, 'input_feature_names', None)
            self.detach_feature_tensors = getattr(config, 'detach_feature_tensors', True)
        else:
            # Backward compatibility for subclasses that don't pass config
            self.pass_input_features = False
            self.input_feature_names = None
            self.detach_feature_tensors = True
    
    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        feature_tensors: Optional[FeatureTensorDict] = None
    ) -> Tensor:
        """
        Compute loss with optional feature tensor support.
        
        Args:
            predictions: Model predictions [batch_size, ...]
            targets: Ground truth targets [batch_size, ...]
            feature_tensors: Optional dict of input features
                {feature_name: tensor[batch_size, feature_dim]}
                Only provided if pass_input_features=true in config
        
        Returns:
            Loss value (scalar tensor)
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def check_feature_compatibility(self, required_features: Optional[list] = None) -> None:
        """
        Validate that required features are available if this loss needs them.
        Called during model setup to catch configuration errors early.
        
        Args:
            required_features: List of feature names required by this loss
            
        Raises:
            ValueError: If required features are not available
        """
        if required_features and not self.pass_input_features:
            raise ValueError(
                f"{self.__class__.__name__} requires input features {required_features} "
                f"but pass_input_features=false in loss configuration. "
                f"Set pass_input_features=true to enable feature tensor support."
            )
        
        if (required_features and self.input_feature_names and 
            not set(required_features).issubset(set(self.input_feature_names))):
            missing = set(required_features) - set(self.input_feature_names)
            raise ValueError(
                f"{self.__class__.__name__} requires features {missing} "
                f"but they are not in input_feature_names: {self.input_feature_names}"
            )


class LogitsInputsMixin:
    @classmethod
    def get_loss_inputs(cls):
        """Maps loss to the desired predicted input type."""
        return LOGITS


@register_loss(MSELossConfig)
class MSELoss(BaseLoss, LogitsInputsMixin):
    """Mean squared error with optional feature tensor support."""

    def __init__(self, config: MSELossConfig):
        super().__init__(config)
        self.mse_fn = nn.MSELoss()

    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        feature_tensors: Optional[FeatureTensorDict] = None
    ) -> Tensor:
        """
        Compute MSE loss, ignoring feature_tensors (backward compatible).
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets  
            feature_tensors: Ignored for MSE loss (maintains compatibility)
        
        Returns:
            MSE loss value
        """
        return self.mse_fn(predictions, targets)


@register_loss(MAELossConfig)
class MAELoss(BaseLoss, LogitsInputsMixin):
    """Mean absolute error with optional feature tensor support."""

    def __init__(self, config: MAELossConfig):
        super().__init__(config)
        self.mae_fn = nn.L1Loss()

    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        feature_tensors: Optional[FeatureTensorDict] = None
    ) -> Tensor:
        """
        Compute MAE loss, ignoring feature_tensors (backward compatible).
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets  
            feature_tensors: Ignored for MAE loss (maintains compatibility)
        
        Returns:
            MAE loss value
        """
        return self.mae_fn(predictions, targets)


@register_loss(MAPELossConfig)
class MAPELoss(BaseLoss, LogitsInputsMixin):
    """Mean absolute percentage error with optional feature tensor support."""

    def __init__(self, config: MAPELossConfig):
        super().__init__(config)

    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        feature_tensors: Optional[FeatureTensorDict] = None
    ) -> Tensor:
        """
        Compute MAPE loss, ignoring feature_tensors (backward compatible).
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets  
            feature_tensors: Ignored for MAPE loss (maintains compatibility)
        
        Returns:
            MAPE loss value
        """
        return mean_absolute_percentage_error(predictions, targets)


@register_loss(RMSELossConfig)
class RMSELoss(BaseLoss, LogitsInputsMixin):
    """Root mean square error with optional feature tensor support."""

    def __init__(self, config: RMSELossConfig):
        super().__init__(config)
        self.mse = nn.MSELoss()

    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        feature_tensors: Optional[FeatureTensorDict] = None
    ) -> Tensor:
        """
        Compute RMSE loss, ignoring feature_tensors (backward compatible).
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets  
            feature_tensors: Ignored for RMSE loss (maintains compatibility)
        
        Returns:
            RMSE loss value
        """
        return torch.sqrt(self.mse(predictions, targets))


@register_loss(RMSPELossConfig)
class RMSPELoss(BaseLoss, LogitsInputsMixin):
    """Root mean square percentage error with optional feature tensor support."""

    def __init__(self, config: RMSPELossConfig):
        super().__init__(config)

    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        feature_tensors: Optional[FeatureTensorDict] = None
    ) -> Tensor:
        """
        Compute RMSPE loss, ignoring feature_tensors (backward compatible).
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets  
            feature_tensors: Ignored for RMSPE loss (maintains compatibility)
        
        Returns:
            RMSPE loss value
        """
        return utils.rmspe_loss(targets, predictions)


@register_loss(BWCEWLossConfig)
class BWCEWLoss(BaseLoss, LogitsInputsMixin):
    """Binary weighted cross entropy loss with optional feature tensor support."""

    def __init__(self, config: BWCEWLossConfig):
        super().__init__(config)
        if config.positive_class_weight:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([config.positive_class_weight]))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=config.positive_class_weight)
        self.robust_lambda = config.robust_lambda
        self.confidence_penalty = config.confidence_penalty

    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        feature_tensors: Optional[FeatureTensorDict] = None
    ) -> Tensor:
        """
        Compute binary weighted cross-entropy loss, ignoring feature_tensors (backward compatible).
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets  
            feature_tensors: Ignored for BWCEW loss (maintains compatibility)
        
        Returns:
            BWCEW loss value
        """
        train_loss = self.loss_fn(predictions, targets.float())
        # robust lambda
        if self.robust_lambda > 0:
            train_loss = (1 - self.robust_lambda) * train_loss + self.robust_lambda / 2

        train_mean_loss = torch.mean(train_loss)

        # confidence penalty
        if self.confidence_penalty > 0:
            probabilities = torch.sigmoid(predictions)
            mean_penalty = utils.mean_confidence_penalty(probabilities, 2)
            train_mean_loss += self.confidence_penalty * mean_penalty

        return train_mean_loss


@register_loss(SoftmaxCrossEntropyLossConfig)
class SoftmaxCrossEntropyLoss(BaseLoss, LogitsInputsMixin):
    """Softmax cross-entropy loss with optional feature tensor support."""
    
    def __init__(self, config: SoftmaxCrossEntropyLossConfig):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__(config)
        if config.class_weights:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(config.class_weights))
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        feature_tensors: Optional[FeatureTensorDict] = None
    ) -> Tensor:
        """
        Compute softmax cross-entropy loss, ignoring feature_tensors (backward compatible).
        
        Args:
            predictions: Tensor of shape [batch x num_classes]
                          or shape [batch x num_classes x H x W]
            targets: Tensor of shape [batch], where each element is integral
                between 0 and num_classes.
                           or shape [batch x H x W], where each element is integral
                between 0 and num_classes.
            feature_tensors: Ignored for cross-entropy loss (maintains compatibility)
        
        Returns:
            Cross-entropy loss value
        """
        if len(targets.shape) == 1 or len(targets.shape) == 3:
            # Assumes we are providing the target as a single class, rather than a distribution
            # The target shape can be a 3D tensor [batch x H x W], for image segmentation
            targets = targets.long()
        return self.loss_fn(predictions, targets)


@register_loss(SequenceSoftmaxCrossEntropyLossConfig)
class SequenceSoftmaxCrossEntropyLoss(BaseLoss, LogitsInputsMixin):
    """Sequence softmax cross-entropy loss with optional feature tensor support."""
    
    def __init__(self, config: SequenceSoftmaxCrossEntropyLossConfig):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__(config)
        if config.class_weights:
            self.loss_fn = nn.CrossEntropyLoss(
                weight=torch.Tensor(config.class_weights), ignore_index=strings_utils.SpecialSymbol.PADDING.value
            )
        else:
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=strings_utils.SpecialSymbol.PADDING.value)

    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        feature_tensors: Optional[FeatureTensorDict] = None
    ) -> Tensor:
        """
        Compute sequence softmax cross-entropy loss, ignoring feature_tensors (backward compatible).
        
        Args:
            predictions: Tensor of shape [batch x sequence_length x vocab_size]
            targets: Tensor of shape [batch x sequence_length], where each element is integral between 0 and vocab_size.
            feature_tensors: Ignored for sequence loss (maintains compatibility)
        
        Returns:
            Sequence cross-entropy loss value
        """
        targets = targets.long()
        return self.loss_fn(predictions[1:].view(-1, predictions.size(-1)), targets[1:].view(-1))


@register_loss(NextTokenSoftmaxCrossEntropyLossConfig)
class NextTokenSoftmaxCrossEntropyLoss(BaseLoss, LogitsInputsMixin):
    """Next token softmax cross-entropy loss with optional feature tensor support."""
    
    def __init__(self, config: NextTokenSoftmaxCrossEntropyLossConfig):
        super().__init__(config)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        feature_tensors: Optional[FeatureTensorDict] = None
    ) -> Tensor:
        """
        Compute next token cross-entropy loss, ignoring feature_tensors (backward compatible).
        
        Args:
            predictions: Tensor of shape [batch x sequence_length x vocab_size]
            targets: Tensor of shape [batch x sequence_length], where each element is integral between 0 and vocab_size.
            feature_tensors: Ignored for next token loss (maintains compatibility)

        Returns:
            Next token cross-entropy loss value
            
        Reference implementation:
        https://github.com/huggingface/transformers/blob/v4.29.1/src/transformers/models/bert/modeling_bert.py#LL1253C1-L1260C1 # noqa
        """
        targets = targets.long()
        _, _, vocab_size = predictions.shape
        # logits for all tensors except n+1 since each logit tensor at position i represents the log probabilities for
        # the next token i+1 if we were to do argmax on the logits ensor at position i.
        shifted_predictions = predictions[:, :-1, :]
        # Shift by 1 since the logits at position 0 in predictions represent the log likelihood of target token 1
        shifted_targets = targets[:, 1:]
        return self.loss_fn(shifted_predictions.reshape(-1, vocab_size), shifted_targets.reshape(-1))


@register_loss(SigmoidCrossEntropyLossConfig)
class SigmoidCrossEntropyLoss(BaseLoss, LogitsInputsMixin):
    """Sigmoid cross-entropy loss with optional feature tensor support."""
    
    def __init__(self, config: SigmoidCrossEntropyLossConfig):
        """
        Params:
            class_weights: List or 1D tensor of length equal to number of classes.
        """
        super().__init__(config)
        if config.class_weights:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(config.class_weights))
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        feature_tensors: Optional[FeatureTensorDict] = None
    ) -> Tensor:
        """
        Compute sigmoid cross-entropy loss, ignoring feature_tensors (backward compatible).
        
        Args:
            predictions: Model predictions (must be 2D tensor)
            targets: Ground truth targets
            feature_tensors: Ignored for sigmoid loss (maintains compatibility)
        
        Returns:
            Sigmoid cross-entropy loss value
        """
        if predictions.ndim != 2:
            raise RuntimeError("SigmoidCrossEntropyLoss currently only supported for 2D tensors.")

        return self.loss_fn(predictions.type(torch.float32), targets.type(torch.float32))


@register_loss(HuberLossConfig)
class HuberLoss(BaseLoss, LogitsInputsMixin):
    """Huber loss with optional feature tensor support."""

    def __init__(self, config: HuberLossConfig):
        super().__init__(config)
        self.huber_fn = _HuberLoss(delta=config.delta)

    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        feature_tensors: Optional[FeatureTensorDict] = None
    ) -> Tensor:
        """
        Compute Huber loss, ignoring feature_tensors (backward compatible).
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            feature_tensors: Ignored for Huber loss (maintains compatibility)
        
        Returns:
            Huber loss value
        """
        return self.huber_fn(predictions, targets)


@register_loss(CORNLossConfig)
class CORNLoss(BaseLoss, LogitsInputsMixin):
    """CORN loss with optional feature tensor support."""

    def __init__(self, config: CORNLossConfig):
        super().__init__(config)

    def forward(
        self, 
        predictions: Tensor, 
        targets: Tensor,
        feature_tensors: Optional[FeatureTensorDict] = None
    ) -> Tensor:
        """
        Compute CORN loss, ignoring feature_tensors (backward compatible).
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            feature_tensors: Ignored for CORN loss (maintains compatibility)
        
        Returns:
            CORN loss value
        """
        num_classes = predictions.shape[1]
        return corn_loss(predictions, targets, num_classes=num_classes)
