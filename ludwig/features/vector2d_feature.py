import torch
import numpy as np
from collections import Counter
import logging
import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torchvision.transforms import functional as F
from torchvision.transforms.functional import normalize
from functools import partial
from ludwig.data.cache.types import wrap
from ludwig.constants import (
    CHECKSUM,
    COLUMN,
    ENCODER,
    HEIGHT,
    HIDDEN,
    IMAGE,
    IMAGENET1K,
    INFER_IMAGE_DIMENSIONS,
    INFER_IMAGE_MAX_HEIGHT,
    INFER_IMAGE_MAX_WIDTH,
    INFER_IMAGE_SAMPLE_SIZE,
    NAME,
    NUM_CHANNELS,
    PREPROCESSING,
    PROC_COLUMN,
    REQUIRES_EQUAL_DIMENSIONS,
    SRC,
    TRAINING,
    TYPE,
    WIDTH, VECTOR2D, PREDICTIONS, LOGITS, MODEL_ECD, MEAN_SQUARED_ERROR,
)
from ludwig.features.base_feature import InputFeature, OutputFeature, BaseFeatureMixin, PredictModule
from ludwig.features.image_feature import ImageFeatureMixin, ImageTransformMetadata
from ludwig.schema.features.vector2d_feature import  Vector2DOutputFeatureConfig
from ludwig.types import PreprocessingConfigDict, FeatureMetadataDict, TrainingSetMetadataDict, \
    FeaturePostProcessingOutputDict
from ludwig.utils import output_feature_utils
from ludwig.utils.image_utils import (
    get_gray_default_image,
    grayscale,
    is_torchvision_encoder,
    num_channels_in_image,
    read_image_from_bytes_obj,
    read_image_from_path,
    resize_image,
    ResizeChannels,
    torchvision_model_registry, TVModelVariant,
)

from ludwig.utils.data_utils import get_abs_path
from ludwig.utils.dataframe_utils import is_dask_series_or_df
from ludwig.utils.fs_utils import has_remote_protocol, upload_h5
from ludwig.utils.misc_utils import set_default_value
from ludwig.utils.types import Series, TorchscriptPreprocessingInput

logger = logging.getLogger(__name__)

def _get_torchvision_transform(
    torchvision_parameters: TVModelVariant,
) -> Tuple[torch.nn.Module, ImageTransformMetadata]:
    """Returns a torchvision transform that is compatible with the model variant.

    Note that the raw torchvision transform is not returned. Instead, a Sequential module that includes
    image resizing is returned. This is because the raw torchvision transform assumes that the input image has
    three channels, which is not always the case with images input into Ludwig.

    Args:
        torchvision_parameters: The parameters for the torchvision model variant.
    Returns:
        (torchvision_transform, transform_metadata): A torchvision transform and the metadata for the transform.
    """
    torchvision_transform_raw = torchvision_parameters.model_weights.DEFAULT.transforms()
    torchvision_transform = torch.nn.Sequential(
        ResizeChannels(num_channels=3),
        torchvision_transform_raw,
    )
    transform_metadata = ImageTransformMetadata(
        height=torchvision_transform_raw.crop_size[0],
        width=torchvision_transform_raw.crop_size[0],
        num_channels=len(torchvision_transform_raw.mean),
    )
    return (torchvision_transform, transform_metadata)


def _get_torchvision_parameters(model_type: str, model_variant: str) -> TVModelVariant:
    return torchvision_model_registry.get(model_type).get(model_variant)

class Vector2DFeatureMixin(BaseFeatureMixin):
    @staticmethod
    def type():
        return VECTOR2D

    @staticmethod
    def cast_column(column, backend):
        return column

    @staticmethod
    def get_feature_meta(
            column, preprocessing_parameters: PreprocessingConfigDict, backend, is_input_feature: bool
    ) -> FeatureMetadataDict:
        return {PREPROCESSING: preprocessing_parameters}

    @staticmethod
    def _read_image_if_bytes_obj_and_resize(
            img_entry: Union[bytes, torch.Tensor, np.ndarray],
            img_width: int,
            img_height: int,
            should_resize: bool,
            num_channels: int,
            resize_method: str,
            user_specified_num_channels: bool,
            standardize_image: str,
    ) -> Optional[np.ndarray]:
        """
        :param img_entry Union[bytes, torch.Tensor, np.ndarray]: if str file path to the
            image else torch.Tensor of the image itself
        :param img_width: expected width of the image
        :param img_height: expected height of the image
        :param should_resize: Should the image be resized?
        :param resize_method: type of resizing method
        :param num_channels: expected number of channels in the first image
        :param user_specified_num_channels: did the user specify num channels?
        :param standardize_image: specifies whether to standarize image with imagenet1k specifications
        :return: image object as a numpy array

        Helper method to read and resize an image according to model definition.
        If the user doesn't specify a number of channels, we use the first image
        in the dataset as the source of truth. If any image in the dataset
        doesn't have the same number of channels as the first image,
        raise an exception.

        If the user specifies a number of channels, we try to convert all the
        images to the specifications by dropping channels/padding 0 channels
        """

        if isinstance(img_entry, bytes):
            img = read_image_from_bytes_obj(img_entry, num_channels)
        elif isinstance(img_entry, str):
            img = read_image_from_path(img_entry, num_channels)
        elif isinstance(img_entry, np.ndarray):
            img = torch.from_numpy(np.array(img_entry, copy=True)).permute(2, 0, 1)
        else:
            img = img_entry

        if not isinstance(img, torch.Tensor):
            warnings.warn(f"Image with value {img} cannot be read")
            return None

        img_num_channels = num_channels_in_image(img)

        vec_2d_size = img_height * img_width

        assert img_num_channels == 1
        # Convert to grayscale if needed.
        if num_channels == 1 and img_num_channels != 1:
            img = grayscale(img)
            img_num_channels = 1

        if should_resize:
            img = resize_image(img, (img_height, img_width), resize_method)

        if user_specified_num_channels:
            # Number of channels is specified by the user
            # img_padded = np.zeros((img_height, img_width, num_channels),
            #                       dtype=np.uint8)
            # min_num_channels = min(num_channels, img_num_channels)
            # img_padded[:, :, :min_num_channels] = img[:, :, :min_num_channels]
            # img = img_padded
            if num_channels > img_num_channels:
                extra_channels = num_channels - img_num_channels
                img = torch.nn.functional.pad(img, [0, 0, 0, 0, 0, extra_channels])

            if img_num_channels != num_channels:
                logger.warning(
                    "Image has {} channels, where as {} "
                    "channels are expected. Dropping/adding channels "
                    "with 0s as appropriate".format(img_num_channels, num_channels)
                )
        else:
            # If the image isn't like the first image, raise exception
            if img_num_channels != num_channels:
                raise ValueError(
                    "Image has {} channels, unlike the first image, which "
                    "has {} channels. Make sure all the images have the same "
                    "number of channels or use the num_channels property in "
                    "image preprocessing".format(img_num_channels, num_channels)
                )

        if img.shape[1] != img_height or img.shape[2] != img_width:
            raise ValueError(
                "Images are not of the same size. "
                "Expected size is {}, "
                "current image size is {}."
                "Images are expected to be all of the same size "
                "or explicit image width and height are expected "
                "to be provided. "
                "Additional information: "
                "https://ludwig-ai.github.io/ludwig-docs/latest/configuration/features/image_features"
                "#image-features-preprocessing".format([img_height, img_width, num_channels], img.shape)
            )

        # casting and rescaling
        img = img.type(torch.float32) / 255

        return img.numpy()

    @staticmethod
    def _read_image_with_pretrained_transform(
            img_entry: Union[bytes, torch.Tensor, np.ndarray],
            transform_fn: Callable,
    ) -> Optional[np.ndarray]:
        if isinstance(img_entry, bytes):
            img = read_image_from_bytes_obj(img_entry)
        elif isinstance(img_entry, str):
            img = read_image_from_path(img_entry)
        elif isinstance(img_entry, np.ndarray):
            img = torch.from_numpy(img_entry).permute(2, 0, 1)
        else:
            img = img_entry

        if not isinstance(img, torch.Tensor):
            warnings.warn(f"Image with value {img} cannot be read")
            return None

        img = transform_fn(img)

        return img.numpy()

    @staticmethod
    def _set_image_and_height_equal_for_encoder(
            width: int, height: int, preprocessing_parameters: dict, encoder_type: str
    ) -> Tuple[int, int]:
        """Some pretrained image encoders require images with the same dimension, or images with a specific width
        and heigh values. The returned width and height are set based on compatibility with the downstream encoder
        using the encoder parameters for the feature.

        Args:
            width: Represents the width of the image. This is either specified in the user config, or inferred using
                a sample of images.
            height: Represents the height of the image. This is either specified in the user config, or inferred using
                a sample of images.
            preprocessing_parameters: Parameters defining how the image feature should be preprocessed
            encoder_type: The name of the encoder

        Return:
            (width, height) Updated width and height so that they are equal
        """

        if preprocessing_parameters[REQUIRES_EQUAL_DIMENSIONS] and height != width:
            width = height = min(width, height)
            # Update preprocessing parameters dictionary to reflect new height and width values
            preprocessing_parameters["width"] = width
            preprocessing_parameters["height"] = height
            logger.info(
                f"Set image feature height and width to {width} to be compatible with" f" {encoder_type} encoder."
            )
        return width, height

    @staticmethod
    def _infer_image_size(
            image_sample: List[torch.Tensor],
            max_height: int,
            max_width: int,
            preprocessing_parameters: dict,
            encoder_type: str,
    ) -> Tuple[int, int]:
        """Infers the size to use from a group of images. The returned height will be the average height of images
        in image_sample rounded to the nearest integer, or max_height. Likewise for width.

        Args:
            image_sample: Sample of images to use to infer image size. Must be formatted as [channels, height, width].
            max_height: Maximum height.
            max_width: Maximum width.
            preprocessing_parameters: Parameters defining how the image feature should be preprocessed
            encoder_type: The name of the encoder

        Return:
            (height, width) The inferred height and width.
        """

        height_avg = sum(x.shape[1] for x in image_sample) / len(image_sample)
        width_avg = sum(x.shape[2] for x in image_sample) / len(image_sample)
        height = min(int(round(height_avg)), max_height)
        width = min(int(round(width_avg)), max_width)

        # Update height and width if the downstream encoder requires images
        # with  the same dimension or specific width and height values
        width, height = ImageFeatureMixin._set_image_and_height_equal_for_encoder(
            width, height, preprocessing_parameters, encoder_type
        )

        logger.debug(f"Inferring height: {height} and width: {width}")
        return height, width

    @staticmethod
    def _infer_number_of_channels(image_sample: List[torch.Tensor]):
        """Infers the channel depth to use from a group of images.

        We make the assumption that the majority of datasets scraped from the web will be RGB, so if we get a mixed bag
        of images we should default to that. However, if the majority of the sample images have a specific channel depth
        (other than 3) this is probably intentional so we keep it, but log an info message.
        """
        n_images = len(image_sample)
        channel_frequency = Counter([num_channels_in_image(x) for x in image_sample])
        if channel_frequency[1] > n_images / 2:
            # If the majority of images in sample are 1 channel, use 1.
            num_channels = 1
        elif channel_frequency[2] > n_images / 2:
            # If the majority of images in sample are 2 channel, use 2.
            num_channels = 2
        elif channel_frequency[4] > n_images / 2:
            # If the majority of images in sample are 4 channel, use 4.
            num_channels = 4
        else:
            # Default case: use 3 channels.
            num_channels = 3
        logger.info(f"Inferring num_channels from the first {n_images} images.")
        logger.info("\n".join([f"  images with {k} channels: {v}" for k, v in sorted(channel_frequency.items())]))
        if num_channels == max(channel_frequency, key=channel_frequency.get):
            logger.info(
                f"Using {num_channels} channels because it is the majority in sample. If an image with"
                f" a different depth is read, will attempt to convert to {num_channels} channels."
            )
        else:
            logger.info(f"Defaulting to {num_channels} channels.")
        logger.info(
            "To explicitly set the number of channels, define num_channels in the preprocessing dictionary of "
            "the image input feature config."
        )
        return num_channels

    @staticmethod
    def _finalize_preprocessing_parameters(
            preprocessing_parameters: dict,
            encoder_type: str,
            column: Series,
    ) -> Tuple:
        """Helper method to determine the height, width and number of channels for preprocessing the image data.

        This is achieved by looking at the parameters provided by the user. When there are some missing parameters, we
        fall back on to the first image in the dataset. The assumption being that all the images in the data are
        expected be of the same size with the same number of channels.

        Args:
            preprocessing_parameters: Parameters defining how the image feature should be preprocessed
            encoder_type: The name of the encoder
            column: The data itself. Can be a Pandas, Modin or Dask series.
        """

        explicit_height_width = preprocessing_parameters[HEIGHT] or preprocessing_parameters[WIDTH]
        explicit_num_channels = NUM_CHANNELS in preprocessing_parameters and preprocessing_parameters[NUM_CHANNELS]

        if preprocessing_parameters[INFER_IMAGE_DIMENSIONS] and not (explicit_height_width and explicit_num_channels):
            sample_size = min(len(column), preprocessing_parameters[INFER_IMAGE_SAMPLE_SIZE])
        else:
            sample_size = 1  # Take first image

        sample = []
        sample_num_bytes = []
        failed_entries = []
        for image_entry in column.head(sample_size):
            if isinstance(image_entry, str):
                # Tries to read image as PNG or numpy file from the path.
                image, num_bytes = read_image_from_path(image_entry, return_num_bytes=True)
                if num_bytes is not None:
                    sample_num_bytes.append(num_bytes)
            else:
                image = image_entry

            if isinstance(image, torch.Tensor):
                sample.append(image)
            elif isinstance(image, np.ndarray):
                sample.append(torch.from_numpy(image).permute(2, 0, 1))
            else:
                failed_entries.append(image_entry)
        if len(sample) == 0:
            failed_entries_repr = "\n\t- ".join(failed_entries)
            raise ValueError(
                f"Images dimensions cannot be inferred. Failed to read {sample_size} images as samples:\n\t- "
                f"{failed_entries_repr}."
            )

        should_resize = False
        if explicit_height_width:
            should_resize = True
            try:
                height = int(preprocessing_parameters[HEIGHT])
                width = int(preprocessing_parameters[WIDTH])
                # Update height and width if the downstream encoder requires images
                # with the same dimension or specific width and height values
                width, height = ImageFeatureMixin._set_image_and_height_equal_for_encoder(
                    width, height, preprocessing_parameters, encoder_type
                )
            except ValueError as e:
                raise ValueError("Image height and width must be set and have " "positive integer values: " + str(e))
            if height <= 0 or width <= 0:
                raise ValueError("Image height and width must be positive integers")
        else:
            # User hasn't specified height and width.
            # Default to inferring from sample or first image.
            if preprocessing_parameters[INFER_IMAGE_DIMENSIONS]:
                should_resize = True
                height, width = ImageFeatureMixin._infer_image_size(
                    sample,
                    max_height=preprocessing_parameters[INFER_IMAGE_MAX_HEIGHT],
                    max_width=preprocessing_parameters[INFER_IMAGE_MAX_WIDTH],
                    preprocessing_parameters=preprocessing_parameters,
                    encoder_type=encoder_type,
                )
            else:
                raise ValueError(
                    "Explicit image width/height are not set, infer_image_dimensions is false, "
                    "and first image cannot be read, so image dimensions are unknown"
                )

        if explicit_num_channels:
            # User specified num_channels in the model/feature config
            user_specified_num_channels = True
            num_channels = preprocessing_parameters[NUM_CHANNELS]
        else:
            user_specified_num_channels = False
            if preprocessing_parameters[INFER_IMAGE_DIMENSIONS]:
                user_specified_num_channels = True
                num_channels = ImageFeatureMixin._infer_number_of_channels(sample)
            elif len(sample) > 0:
                num_channels = num_channels_in_image(sample[0])
            else:
                raise ValueError(
                    "Explicit image num channels is not set, infer_image_dimensions is false, "
                    "and first image cannot be read, so image num channels is unknown"
                )

        assert isinstance(num_channels, int), ValueError("Number of image channels needs to be an integer")

        average_file_size = np.mean(sample_num_bytes) if sample_num_bytes else None

        standardize_image = preprocessing_parameters["standardize_image"]
        if standardize_image == "imagenet1k" and num_channels != 3:
            warnings.warn(
                f"'standardize_image=imagenet1k' is defined only for 'num_channels=3' but "
                f"detected 'num_channels={num_channels}'.  For this situation setting 'standardize_image=None'.",
                RuntimeWarning,
            )
            standardize_image = None

        return (
            should_resize,
            width,
            height,
            num_channels,
            user_specified_num_channels,
            average_file_size,
            standardize_image,
        )

    @staticmethod
    def add_feature_data(
            feature_config,
            input_df,
            proc_df,
            metadata,
            preprocessing_parameters: PreprocessingConfigDict,
            backend,
            skip_save_processed_input,
    ):
        set_default_value(feature_config[PREPROCESSING], "in_memory", preprocessing_parameters["in_memory"])

        name = feature_config[NAME]
        column = input_df[feature_config[COLUMN]]
        encoder_type = feature_config[ENCODER][TYPE]

        src_path = None
        if SRC in metadata:
            src_path = os.path.dirname(os.path.abspath(metadata.get(SRC)))
        abs_path_column = backend.df_engine.map_objects(
            column,
            lambda row: get_abs_path(src_path, row) if isinstance(row, str) and not has_remote_protocol(row) else row,
        )

        # determine if specified encoder is a torchvision model
        model_type = feature_config[ENCODER].get("type", None)
        model_variant = feature_config[ENCODER].get("model_variant")
        if model_variant:
            torchvision_parameters = _get_torchvision_parameters(model_type, model_variant)
        else:
            torchvision_parameters = None

        if torchvision_parameters:
            logger.warning(
                f"Using the transforms specified for the torchvision model {model_type} {model_variant} "
                f"This includes setting the number of channels is 3 and resizing the image to the needs of the model."
            )

            torchvision_transform, transform_metadata = _get_torchvision_transform(torchvision_parameters)

            # torchvision_parameters is not None
            # perform torchvision model transformations
            read_image_if_bytes_obj_and_resize = partial(
                ImageFeatureMixin._read_image_with_pretrained_transform,
                transform_fn=torchvision_transform,
            )
            average_file_size = None

            # save weight specification in preprocessing section
            preprocessing_parameters[
                "torchvision_model_default_weights"
            ] = f"{torchvision_parameters.model_weights.DEFAULT}"

            # add torchvision model id to preprocessing section for torchscript
            preprocessing_parameters["torchvision_model_type"] = model_type
            preprocessing_parameters["torchvision_model_variant"] = model_variant

            # get required setup parameters for in_memory = False processing
            height = transform_metadata.height
            width = transform_metadata.width
            num_channels = transform_metadata.num_channels
        else:
            # torchvision_parameters is None
            # perform Ludwig specified transformations
            (
                should_resize,
                width,
                height,
                num_channels,
                user_specified_num_channels,
                average_file_size,
                standardize_image,
            ) = ImageFeatureMixin._finalize_preprocessing_parameters(
                preprocessing_parameters, encoder_type, abs_path_column
            )

            metadata[name][PREPROCESSING]["height"] = height
            metadata[name][PREPROCESSING]["width"] = width
            metadata[name][PREPROCESSING]["num_channels"] = num_channels

            read_image_if_bytes_obj_and_resize = partial(
                ImageFeatureMixin._read_image_if_bytes_obj_and_resize,
                img_width=width,
                img_height=height,
                should_resize=should_resize,
                num_channels=num_channels,
                resize_method=preprocessing_parameters["resize_method"],
                user_specified_num_channels=user_specified_num_channels,
                standardize_image=standardize_image,
            )

        # TODO: alternatively use get_average_image() for unreachable images
        default_image = get_gray_default_image(num_channels, height, width)
        metadata[name]["reshape"] = (num_channels, height, width)

        in_memory = feature_config[PREPROCESSING]["in_memory"]
        if in_memory or skip_save_processed_input:
            proc_col = backend.read_binary_files(
                abs_path_column, map_fn=read_image_if_bytes_obj_and_resize, file_size=average_file_size
            )

            num_failed_image_reads = (
                proc_col.isna().sum().compute() if is_dask_series_or_df(proc_col, backend) else proc_col.isna().sum()
            )

            proc_col = backend.df_engine.map_objects(
                proc_col, lambda row: default_image if not isinstance(row, np.ndarray) else row
            )

            proc_df[feature_config[PROC_COLUMN]] = proc_col
        else:
            num_images = len(abs_path_column)
            num_failed_image_reads = 0

            data_fp = backend.cache.get_cache_path(wrap(metadata.get(SRC)), metadata.get(CHECKSUM), TRAINING)
            with upload_h5(data_fp) as h5_file:
                # todo future add multiprocessing/multithreading
                image_dataset = h5_file.create_dataset(
                    feature_config[PROC_COLUMN] + "_data", (num_images, num_channels, height, width), dtype=np.float32
                )
                for i, img_entry in enumerate(abs_path_column):
                    res = read_image_if_bytes_obj_and_resize(img_entry)
                    if isinstance(res, np.ndarray):
                        image_dataset[i, :height, :width, :] = res
                    else:
                        logger.warning(f"Failed to read image {img_entry} while preprocessing feature `{name}`. ")
                        image_dataset[i, :height, :width, :] = default_image
                        num_failed_image_reads += 1
                h5_file.flush()

            proc_df[feature_config[PROC_COLUMN]] = np.arange(num_images)

        if num_failed_image_reads > 0:
            logger.warning(
                f"Failed to read {num_failed_image_reads} images while preprocessing feature `{name}`. "
                "Using default image for these rows in the dataset."
            )

        return proc_df


class _Vector2DPreprocessing(torch.nn.Module):
    def forward(self, v: TorchscriptPreprocessingInput) -> torch.Tensor:
        if torch.jit.isinstance(v, torch.Tensor):
            out = v
        elif torch.jit.isinstance(v, List[torch.Tensor]):
            out = torch.stack(v)
        elif torch.jit.isinstance(v, List[str]):
            vectors = []
            for sample in v:
                vector = torch.tensor([float(x) for x in sample.split()], dtype=torch.float32)
                vectors.append(vector)
            out = torch.stack(vectors)
        else:
            raise ValueError(f"Unsupported input: {v}")

        if out.isnan().any():
            raise ValueError("Scripted NaN handling not implemented for Vector feature")
        return out


class _Vector2DPostprocessing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictions_key = PREDICTIONS
        self.logits_key = LOGITS

    def forward(self, preds: Dict[str, torch.Tensor], feature_name: str) -> FeaturePostProcessingOutputDict:
        predictions = output_feature_utils.get_output_feature_tensor(preds, feature_name, self.predictions_key)
        logits = output_feature_utils.get_output_feature_tensor(preds, feature_name, self.logits_key)

        return {self.predictions_key: predictions, self.logits_key: logits}


class _Vector2DPredict(PredictModule):
    def forward(self, inputs: Dict[str, torch.Tensor], feature_name: str) -> Dict[str, torch.Tensor]:
        logits = output_feature_utils.get_output_feature_tensor(inputs, feature_name, self.logits_key)

        return {self.predictions_key: logits, self.logits_key: logits}


# class Vector2DInputFeature(Vector2DFeatureMixin, InputFeature):
#     def __init__(self, input_feature_config: Vector2DInputFeatureConfig, encoder_obj=None, **kwargs):
#         super().__init__(input_feature_config, **kwargs)
#
#         # input_feature_config.encoder.input_size = input_feature_config.encoder.vector_size
#         if encoder_obj:
#             self.encoder_obj = encoder_obj
#         else:
#             self.encoder_obj = self.initialize_encoder(input_feature_config.encoder)
#
#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         assert isinstance(inputs, torch.Tensor)
#         assert inputs.dtype in [torch.float32, torch.float64]
#         assert len(inputs.shape) == 2
#
#         inputs_encoded = self.encoder_obj(inputs)
#
#         return inputs_encoded
#
#     @property
#     def input_shape(self) -> torch.Size:
#         return torch.Size([self.encoder_obj.config.input_size])
#
#     @property
#     def output_shape(self) -> torch.Size:
#         return self.encoder_obj.output_shape
#
#     @staticmethod
#     def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
#         feature_config.encoder.input_size = feature_metadata["vector_size"]
#
#     @staticmethod
#     def create_preproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
#         return _Vector2DPreprocessing()
#
#     @staticmethod
#     def get_schema_cls():
#         return Vector2DInputFeatureConfig


class Vector2DOutputFeature(Vector2DFeatureMixin, OutputFeature):

    def __init__(
            self,
            output_feature_config: Union[Vector2DOutputFeatureConfig, Dict],
            output_features: Dict[str, OutputFeature],
            **kwargs,
    ):
        self.vector_size = output_feature_config.vector_size
        super().__init__(output_feature_config, output_features, **kwargs)
        # output_feature_config.decoder.output_size = self.vector_size
        output_feature_config.decoder.height = self.height
        output_feature_config.decoder.width = self.width

        self.decoder_obj = self.initialize_decoder(output_feature_config.decoder)
        self._setup_loss()
        self._setup_metrics()

    def logits(self, inputs, **kwargs):  # hidden
        hidden = inputs[HIDDEN]
        return self.decoder_obj(hidden)

    def metric_kwargs(self):
        return dict(num_outputs=self.output_shape[0])

    def create_predict_module(self) -> PredictModule:
        return _Vector2DPredict()

    def get_prediction_set(self):
        return {PREDICTIONS, LOGITS}

    @classmethod
    def get_output_dtype(cls):
        return torch.float32

    @property
    def output_shape(self) -> torch.Size:
        return torch.Size([self.vector_size])

    @property
    def input_shape(self) -> torch.Size:
        return torch.Size([self.input_size])

    # @staticmethod
    # def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
    #    feature_config.vector_size = feature_metadata["vector_size"]

    @staticmethod
    def update_config_with_metadata(feature_config, feature_metadata, *args, **kwargs):
        for key in ["height", "width", "num_channels", "standardize_image"]:
            if hasattr(feature_config.decoder, key):
                setattr(feature_config.decoder, key, feature_metadata[PREPROCESSING][key])

    @staticmethod
    def calculate_overall_stats(predictions, targets, train_set_metadata):
        # no overall stats, just return empty dictionary
        return {}

    def postprocess_predictions(
            self,
            result,
            metadata,
    ):
        predictions_col = f"{self.feature_name}_{PREDICTIONS}"
        if predictions_col in result:
            result[predictions_col] = result[predictions_col].map(lambda pred: pred.tolist())
        return result

    @staticmethod
    def create_postproc_module(metadata: TrainingSetMetadataDict) -> torch.nn.Module:
        return _Vector2DPostprocessing()

    @staticmethod
    def get_schema_cls():
        return Vector2DOutputFeatureConfig

    # add a new decoder similar to projector and then modify it for 2DVec


