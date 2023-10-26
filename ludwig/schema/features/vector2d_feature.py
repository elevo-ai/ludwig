from ludwig.constants import VECTOR2D, MODEL_ECD, MEAN_SQUARED_ERROR
from ludwig.schema.decoders.base import BaseDecoderConfig
from ludwig.schema.decoders.utils import DecoderDataclassField
from ludwig.schema.encoders.base import BaseEncoderConfig
from ludwig.schema.encoders.utils import EncoderDataclassField
from ludwig.schema.features.base import BaseInputFeatureConfig, BaseOutputFeatureConfig
from ludwig.schema.features.loss.loss import BaseLossConfig
from ludwig.schema.features.loss.utils import LossDataclassField
from ludwig.schema.features.preprocessing.base import BasePreprocessingConfig
from ludwig.schema.features.preprocessing.utils import PreprocessingDataclassField, register_preprocessor
from ludwig.schema.features.utils import input_mixin_registry, ecd_input_config_registry, output_mixin_registry, \
    ecd_output_config_registry
from ludwig.schema.metadata import FEATURE_METADATA
from ludwig.schema.metadata.parameter_metadata import INTERNAL_ONLY
from ludwig.schema.utils import ludwig_dataclass, BaseMarshmallowConfig
from ludwig.schema import utils as schema_utils
from ludwig.api_annotations import DeveloperAPI


# @DeveloperAPI
# @input_mixin_registry.register(VECTOR2D)
# @ludwig_dataclass
# class Vector2DInputFeatureConfigMixin(BaseMarshmallowConfig):
#     """VectorInputFeatureConfigMixin is a dataclass that configures the parameters used in both the vector input
#     feature and the vector global defaults section of the Ludwig Config."""
#
#     #preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type=VECTOR2D)
#
#     encoder: BaseEncoderConfig = EncoderDataclassField(
#         MODEL_ECD,
#         feature_type=VECTOR2D,
#         default="dense",
#     )
#
#
# @DeveloperAPI
# @ecd_input_config_registry.register(VECTOR2D)
# @ludwig_dataclass
# class Vector2DInputFeatureConfig(Vector2DInputFeatureConfigMixin, BaseInputFeatureConfig):
#     """VectorInputFeatureConfig is a dataclass that configures the parameters used for a vector input feature."""
#
#     type: str = schema_utils.ProtectedString(VECTOR2D)


@DeveloperAPI
@output_mixin_registry.register(VECTOR2D)
@ludwig_dataclass
class Vector2DOutputFeatureConfigMixin(BaseMarshmallowConfig):
    """VectorOutputFeatureConfigMixin is a dataclass that configures the parameters used in both the vector output
    feature and the vector global defaults section of the Ludwig Config."""

    decoder: BaseDecoderConfig = DecoderDataclassField(
        MODEL_ECD,
        feature_type=VECTOR2D,
        default="projector2d",
    )

    loss: BaseLossConfig = LossDataclassField(
        feature_type=VECTOR2D,
        default=MEAN_SQUARED_ERROR,
    )


@DeveloperAPI
@ecd_output_config_registry.register(VECTOR2D)
#@register_preprocessor(VECTOR2D)
@ludwig_dataclass
class Vector2DOutputFeatureConfig(Vector2DOutputFeatureConfigMixin, BaseOutputFeatureConfig):
    """VectorOutputFeatureConfig is a dataclass that configures the parameters used for a vector output feature."""

    preprocessing: BasePreprocessingConfig = PreprocessingDataclassField(feature_type="vector2d")

    type: str = schema_utils.ProtectedString(VECTOR2D)

    dependencies: list = schema_utils.List(
        default=[],
        description="List of input features that this feature depends on.",
        parameter_metadata=FEATURE_METADATA[VECTOR2D]["dependencies"],
    )

    default_validation_metric: str = schema_utils.StringOptions(
        [MEAN_SQUARED_ERROR],
        default=MEAN_SQUARED_ERROR,
        description="Internal only use parameter: default validation metric for binary output feature.",
        parameter_metadata=INTERNAL_ONLY,
    )



    reduce_dependencies: str = schema_utils.ReductionOptions(
        default=None,
        description="How to reduce the dependencies of the output feature.",
        parameter_metadata=FEATURE_METADATA[VECTOR2D]["reduce_dependencies"],
    )

    reduce_input: str = schema_utils.ReductionOptions(
        default=None,
        description="How to reduce an input that is not a vector, but a matrix or a higher order tensor, on the first "
        "dimension (second if you count the batch dimension)",
        parameter_metadata=FEATURE_METADATA[VECTOR2D]["reduce_input"],
    )

    softmax: bool = schema_utils.Boolean(
        default=False,
        description="Determines whether to apply a softmax at the end of the decoder. This is useful for predicting a "
        "vector of values that sum up to 1 and can be interpreted as probabilities.",
        parameter_metadata=FEATURE_METADATA[VECTOR2D]["softmax"],
    )

    vector_size: int = schema_utils.PositiveInteger(
        default=None,
        allow_none=True,
        description="The size of the vector. If None, the vector size will be inferred from the data.",
        parameter_metadata=FEATURE_METADATA[VECTOR2D]["vector_size"],
    )