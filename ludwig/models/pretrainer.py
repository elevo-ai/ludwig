import logging
import math

import numpy as np
import torch
from pytorch_metric_learning.losses import NTXentLoss
from typing import Dict, Any

from ludwig.data.dataset.base import Dataset
from ludwig.models.ecd import ECD
from ludwig.models.trainer import Trainer
from ludwig.modules.fully_connected_modules import FCStack
from ludwig.utils.torch_utils import LudwigModule, Dense

logger = logging.getLogger(__name__)


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


class SimCLR_Loss(torch.nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = torch.nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        # SIMCLR
        labels = torch.from_numpy(np.array([0] * N)).reshape(-1).to(positive_samples.device).long()  # .float()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class LinearLayer(torch.nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias=True,
                 use_bn=False,
                 **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn

        self.linear = torch.nn.Linear(self.in_features,
                                      self.out_features,
                                      bias=self.use_bias and not self.use_bn)
        if self.use_bn:
            self.bn = torch.nn.BatchNorm1d(self.out_features)

    def forward(self, x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class ProjectionHead(torch.nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 head_type='nonlinear',
                 **kwargs):
        super(ProjectionHead, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features, self.out_features, False, True)
        elif self.head_type == 'nonlinear':
            self.layers = torch.nn.Sequential(
                LinearLayer(self.in_features, self.hidden_features, True, True),
                torch.nn.ReLU(),
                LinearLayer(self.hidden_features, self.out_features, False, True))

    def forward(self, x):
        x = self.layers(x)
        return x


class ScarfModel(LudwigModule):
    def __init__(
            self,
            model: ECD,
            training_set_metadata: Dict[str, Any],
            corruption_rate: float = 0.6,
            temperature: float = 1.0,
    ):
        super().__init__()
        self.training_set_metadata = training_set_metadata
        self.input_features = model.input_features
        self.output_features = torch.nn.ModuleDict()
        self.combiner = model.combiner
        self.projection_head = ProjectionHead(
            self.combiner.output_shape[-1],
            256,
            64
        )
        self.num_corrupted_features = math.floor(corruption_rate * len(self.input_features))
        # self.loss_fn = NTXentLoss(temperature=temperature)
        self.loss_fn = SimCLR_Loss(128, temperature)

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            inputs, _ = inputs

        assert inputs.keys() == self.input_features.keys()
        for input_feature_name, input_values in inputs.items():
            inputs[input_feature_name] = torch.from_numpy(input_values)

        anchor_embeddings = self._embed(inputs)
        corrupted_embeddings = self._embed(self._corrupt(inputs))
        return anchor_embeddings, corrupted_embeddings

    def _corrupt(self, inputs):
        # per SCARF paper: select a subset of the features and replace them with a
        # sample from the marginal training distribution

        # compute augmentations for all features
        batch_size = None
        augmentations = {}
        for input_feature_name, input_values in inputs.items():
            batch_size = len(input_values)
            encoder = self.input_features[input_feature_name]
            augmentations[input_feature_name] = encoder.sample_augmentations(
                batch_size,
                self.training_set_metadata[input_feature_name]
            )

        # construct N x M matrix, where every row (batch) samples q features
        # to corrupt
        m = len(inputs)
        mask = np.zeros((batch_size, m), dtype=int)
        mask[:, :self.num_corrupted_features] = 1
        mask = shuffle_along_axis(mask, axis=1)

        corrupted_inputs = {
            input_feature_name: torch.from_numpy(np.where(
                mask[:, j],
                augmentations[input_feature_name],
                input_values
            ))
            for j, (input_feature_name, input_values) in enumerate(inputs.items())
        }

        return corrupted_inputs

    def _embed(self, inputs):
        encoder_outputs = {}
        for input_feature_name, input_values in inputs.items():
            encoder = self.input_features[input_feature_name]
            encoder_output = encoder(input_values)
            encoder_outputs[input_feature_name] = encoder_output

        combiner_outputs = self.combiner(encoder_outputs)
        return self.projection_head(combiner_outputs['combiner_output'])

    def train_loss(self, targets, predictions, regularization_lambda=0.0):
        print(predictions)
        anchor_embeddings, corrupted_embeddings = predictions
        embeddings = torch.cat((anchor_embeddings, corrupted_embeddings))
        indices = torch.arange(0, anchor_embeddings.size(0), device=anchor_embeddings.device)
        labels = torch.cat((indices, indices))
        # return self.loss_fn(embeddings, labels), {}
        return self.loss_fn(anchor_embeddings, corrupted_embeddings), {}

    def reset_metrics(self):
        pass


class Pretrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def pretrain(
            self,
            model: ECD,
            dataset: Dataset,
            training_set_metadata: Dict[str, Any],
            **kwargs
    ):
        ssl_model = ScarfModel(model, training_set_metadata)
        _, train_stats, _, _ = self.train(
            ssl_model,
            training_set=dataset,
            **kwargs
        )
        return model, train_stats

    def evaluation(
            self,
            model,
            dataset,
            dataset_name,
            metrics_log,
            tables,
            batch_size=128,
    ):
        pass
