from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch

try:
    import dask.dataframe as dd

    DataFrame = Union[pd.DataFrame, dd.DataFrame]
    Series = Union[pd.Series, dd.Series]
except ImportError:
    DataFrame = pd.DataFrame
    Series = pd.Series

# torchaudio.load returns the audio tensor and the sampling rate as a tuple.
TorchAudioTuple = Tuple[torch.Tensor, int]
TorchscriptPreprocessingInput = Union[List[str], List[torch.Tensor], List[TorchAudioTuple], torch.Tensor]
TorchDevice = Union[str, torch.device]

# Feature Tensors Support Types
# Type alias for feature tensor dictionary passed to loss functions
FeatureTensorDict = Dict[str, torch.Tensor]

# Type alias for loss function with optional feature tensors
LossFunction = Callable[
    [torch.Tensor, torch.Tensor, Optional[FeatureTensorDict]], 
    torch.Tensor
]
