import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor

from lakitu.training.helpers.config import BaseConfig
from lakitu.training.diffusion.policy import PolicyFeature, FeatureType, NormalizationMode, DType

@dataclass
class DistanceEstimatorConfig(BaseConfig):
    device: str = "cuda"

    # Vision backbone
    vision_backbone: str = "resnet18"
    vision_backbone_weights: str | None = None
    vision_features: int = 1024
    crop_shape: tuple[int, int] | None = (240, 180)
    crop_is_random: bool = True
    use_group_norm: bool = True

    # Input / output structure
    n_obs_steps: int = 1
    min_distance: int = 31
    input_features: dict[str, PolicyFeature] = field(default_factory=lambda: {
        "observation.image": PolicyFeature(
            type=FeatureType.VISUAL, shape=(3, 240, 320), dtype=DType.FLOAT, norm_mode=NormalizationMode.IDENTITY
        ),
    })
    output_features: dict[str, PolicyFeature] = field(default_factory=lambda: {
        "info.distance": PolicyFeature(type=FeatureType.STATE, shape=(7,), dtype=DType.FLOAT, norm_mode=NormalizationMode.IDENTITY),
    })

    @property
    def state_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.STATE}

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def output_size(self) -> int:
        return sum(ft.shape[0] for ft in self.output_features.values())


# Helper functions

def symlog(x: Tensor) -> Tensor:
    """Symmetric log function, as described in the DreamerV3 paper"""
    return torch.sign(x) * torch.log2(torch.abs(x) + 1)

def symexp(x: Tensor) -> Tensor:
    """Symmetric exponential function, as described in the DreamerV3 paper"""
    return torch.sign(x) * (torch.exp2(torch.abs(x)) - 1)

def scalar_to_class_probs(x: Tensor, min: int, max: int) -> Tensor:
    """Two-hot encoding scheme, as described in the DreamerV3 paper"""
    assert min < max, "min must be less than max"
    if x.ndim < 1:
        x = x.unsqueeze(0)

    min_idx = int(symlog(torch.tensor(min)).floor().long().item())
    max_idx = int(symlog(torch.tensor(max)).ceil().long().item())
    num_classes = max_idx - min_idx + 1

    x = x.clamp(min=min, max=max)
    left_idx = symlog(x).floor().long()
    right_idx = (left_idx + 1).clamp(max=max_idx)
    left_val = symexp(left_idx)
    right_weight = (x - left_val) / (left_val + 1)  # linear interp from left_val to right_val
    left_weight = 1 - right_weight

    targets = torch.zeros((*x.shape[:-1], num_classes), device=x.device)
    targets.scatter_(-1, right_idx - min_idx, right_weight)  # scatter right weights first since right_idx might be clamped to left_idx
    targets.scatter_(-1, left_idx - min_idx, left_weight)
    assert torch.all((targets >= 0) & (targets <= 1)), "Class probabilities must be in [0, 1]"
    return targets

def class_probs_to_scalar(x: Tensor, min_distance: int) -> Tensor:
    """Decode class probabilities into a scalar value"""
    num_classes = x.shape[-1]
    min_idx = int(math.log2(min_distance + 1))
    points = symexp(torch.arange(min_idx, min_idx + num_classes, device=x.device).float()).broadcast_to(x.shape)
    return (x * points).sum(dim=-1)

def get_distance_targets(distance: Tensor, min: int, max: int) -> Tensor:
    distance = torch.where(distance >= 0, distance, torch.tensor(max, device=distance.device))
    return scalar_to_class_probs(distance, min=min, max=max)

def replace_batchnorm(module: nn.Module) -> nn.Module:
    """Recursively replace BatchNorm2d with GroupNorm"""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups=child.num_features // 16, num_channels=child.num_features))
        else:
            replace_batchnorm(child)
    return module


class DistanceEstimator(nn.Module):
    def __init__(self, config: DistanceEstimatorConfig):
        super().__init__()
        self.config = config
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        global_cond_dim = sum(feat.shape[0] for feat in self.config.state_features.values())
        global_cond_dim += config.vision_features * len(self.config.image_features)
        global_cond_dim *= config.n_obs_steps

        # Set up backbone
        backbone_model = getattr(torchvision.models, config.vision_backbone)(weights=config.vision_backbone_weights)
        self.backbone: nn.Module = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.vision_backbone_weights:
                raise ValueError("You can't replace BatchNorm in a pretrained model without ruining the weights!")
            self.backbone = replace_batchnorm(self.backbone)

        # Set up pooling and final layers
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        dummy_input = torch.zeros(size=dummy_shape)
        with torch.inference_mode():
            output = self.backbone(dummy_input)
        feature_map_shape = tuple(output.shape)[1:]

        self.pool = nn.AvgPool2d(kernel_size=feature_map_shape[1:])
        self.out = nn.Linear(feature_map_shape[0], config.vision_features)

        self.fc1 = nn.Linear(global_cond_dim, config.vision_features)
        self.fc2 = nn.Linear(config.vision_features, config.output_size)

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        pred = self(batch)
        min_distance = self.config.min_distance
        max_distance = 2 ** (int(math.log2(min_distance + 1)) + self.config.output_size - 1) - 1
        targets = get_distance_targets(batch['info.distance'], min=min_distance, max=max_distance)
        assert pred.shape == targets.shape, f"Prediction shape {pred.shape} does not match target shape {targets.shape}"
        loss = F.cross_entropy(pred, targets)
        return loss, {'pred': pred}

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        assert batch.keys() >= self.config.input_features.keys()
        input_feats = self._compute_input_features(batch)
        x = F.relu(self.fc1(input_feats))
        x = self.fc2(x)
        return x

    def _compute_input_features(self, batch: dict[str, Tensor]) -> Tensor:
        image_feature = next(iter(self.config.image_features.keys()))
        batch_size, n_obs_steps = batch[image_feature].shape[:2]
        input_feats = [batch[key] for key in self.config.state_features]
        if self.config.image_features:
            images = torch.cat([batch[key] for key in self.config.image_features], dim=2)
            img_features = self._compute_vision_features(einops.rearrange(images, "b s ... -> (b s) ..."))
            img_features = einops.rearrange(img_features, "(b s) ... -> b s (...)", b=batch_size, s=n_obs_steps)
            input_feats.append(img_features)

        # Concatenate features then flatten to (B, global_cond_dim)
        return torch.cat(input_feats, dim=-1).flatten(start_dim=1)

    def _compute_vision_features(self, x: Tensor) -> Tensor:
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)

        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = F.relu(self.out(x))
        return x

    @classmethod
    def from_pretrained(cls, checkpoint_dir: Path, **kwargs: Any) -> Self:
        from safetensors.torch import safe_open
        with open(checkpoint_dir / 'train_config.json') as f:
            config = DistanceEstimatorConfig.create(json.load(f)['model'], kwargs)
        model: Self = cls(config).to(config.device)
        with safe_open(checkpoint_dir / 'model.safetensors', framework="pt", device=config.device) as f:
            state_dict = {key: f.get_tensor(key) for key in f.keys()}
        model.load_state_dict(state_dict, strict=True)
        return model


if __name__ == "__main__":
    config = DistanceEstimatorConfig(device="cpu")
    model = DistanceEstimator(config).to(config.device)

    # Create dummy batch and run forward pass
    batch = {
        "observation.image": torch.randn(2, 2, 3, 240, 320).to(config.device),
        "info.distance": torch.randint(0, 2048, (2, 1)).to(config.device),
    }
    pred = model.compute_loss(batch)
