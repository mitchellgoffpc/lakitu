import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

DatasetStats = dict[str, dict[str, Tensor]]

class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"

class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ACTION = "ACTION"

@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]

@dataclass
class DiffusionConfig:
    type: str = "DiffusionPolicy"
    device: str = "cuda"
    use_amp: bool = False

    input_features: dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: dict[str, PolicyFeature] = field(default_factory=dict)

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # The original implementation doesn't sample frames for the last 7 steps,
    # which avoids excessive padding and leads to improved training results.
    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Architecture / modeling.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    use_separate_rgb_encoder_per_camera: bool = False
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    # Unet.
    down_dims: tuple[int, ...] = (512, 1024, 2048)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True
    spatial_softmax_num_keypoints: int = 512
    # Noise scheduler.
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    @property
    def state_feature(self) -> PolicyFeature:
        return next(ft for ft in self.input_features.values() if ft.type is FeatureType.STATE)

    @property
    def action_feature(self) -> PolicyFeature:
        return next(ft for ft in self.output_features.values() if ft.type is FeatureType.ACTION)

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))


# Helper functions

def _make_noise_scheduler(name: str, **kwargs: Any) -> Any:
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")

def _replace_batchnorm(module: nn.Module) -> nn.Module:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups=child.num_features // 16, num_channels=child.num_features))
        else:
            _replace_batchnorm(child)
    return module


class DiffusionPolicy(nn.Module):
    def __init__(self, config: DiffusionConfig, dataset_stats: DatasetStats | None = None):
        super().__init__()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.state_feature.shape[0]
        if self.config.image_features:
            self.rgb_encoder = DiffusionRgbEncoder(config)
            global_cond_dim += self.rgb_encoder.feature_dim * len(self.config.image_features)

        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        self.reset()

    # Inference methods

    def reset(self) -> None:
        self._queues: dict[str, deque[Tensor]] = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = batch | {"observation.images": torch.cat([batch[key] for key in self.config.image_features], dim=1)}

        for key in batch:
            if key in self._queues:
                while len(self._queues[key]) != self._queues[key].maxlen:
                    self._queues[key].append(batch[key])  # initialize by copying the first observation several times until queue is full
                self._queues[key].append(batch[key])  # add latest observation to the queue

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.generate_actions(batch)
            actions = self.unnormalize_outputs({"action": actions})["action"]
            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)  # encode image features and concatenate with state vector
        actions = self.conditional_sample(batch_size, global_cond=global_cond)  # run sampling

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]

    def conditional_sample(self, batch_size: int, global_cond: Tensor) -> Tensor:
        device = torch.device(self.config.device)
        sample = torch.randn(size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]), device=device)
        self.noise_scheduler.set_timesteps(self.config.num_inference_steps or self.config.num_train_timesteps)

        for t in self.noise_scheduler.timesteps:
            timesteps = torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device)
            model_output = self.unet(sample, timesteps, global_cond)
            sample = self.noise_scheduler.step(model_output, t, sample).prev_sample

        return sample

    # Training methods

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = batch | {"observation.images": torch.cat([batch[key] for key in self.config.image_features], dim=1)}
        batch = self.normalize_targets(batch)
        loss, output_dict = self.compute_loss(batch)
        return loss, output_dict


    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion
        trajectory = batch["action"]
        eps = torch.randn(trajectory.shape, device=trajectory.device)  # sample noise
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, size=(trajectory.shape[0],), device=trajectory.device)
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network
        pred = self.unet(noisy_trajectory, timesteps, global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(f"You need to provide 'action_is_pad' in the batch when {self.config.do_mask_loss_for_padding=}.")
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean(), {'pred': pred}

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        global_cond_feats = [batch["observation.state"]]
        if self.config.image_features:
            img_features = self.rgb_encoder(einops.rearrange(batch["observation.images"], "b s ... -> (b s) ..."))
            img_features = einops.rearrange(img_features, "(b s) ... -> b s (...)", b=batch_size, s=n_obs_steps)
            global_cond_feats.append(img_features)

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)


class DiffusionRgbEncoder(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone
        backbone_model = getattr(torchvision.models, config.vision_backbone)(weights=config.pretrained_backbone_weights)
        self.backbone: nn.Module = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("You can't replace BatchNorm in a pretrained model without ruining the weights!")
            self.backbone = _replace_batchnorm(self.backbone)

        # Set up pooling and final layers
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        dummy_input = torch.zeros(size=dummy_shape)
        with torch.inference_mode():
            output = self.backbone(dummy_input)
        feature_map_shape = tuple(output.shape)[1:]

        self.pool = nn.AvgPool2d(kernel_size=feature_map_shape[1:])
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(feature_map_shape[0], self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)

        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x


class DiffusionConditionalUnet1d(nn.Module):
    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()
        self.config = config

        # Encoder for the diffusion timestep
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we just reverse these.
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + \
            list(zip(config.down_dims[:-1], config.down_dims[1:], strict=True))

        # Unet encoder
        kernel_size, n_groups, use_film_scale = config.kernel_size, config.n_groups, config.use_film_scale_modulation
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                DiffusionConditionalResidualBlock1d(dim_in, dim_out, cond_dim, kernel_size, n_groups, use_film_scale),
                DiffusionConditionalResidualBlock1d(dim_out, dim_out, cond_dim, kernel_size, n_groups, use_film_scale),
                # Downsample as long as it is not the last block.
                nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
            ]))

        # Processing in the middle of the auto-encoder
        mid_dim = config.down_dims[-1]
        self.mid_modules = nn.ModuleList([
            DiffusionConditionalResidualBlock1d(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, use_film_scale),
            DiffusionConditionalResidualBlock1d(mid_dim, mid_dim, cond_dim, kernel_size, n_groups, use_film_scale),
        ])

        # Unet decoder
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                # dim_in * 2, because it takes the encoder's skip connection as well
                DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, cond_dim, kernel_size, n_groups, use_film_scale),
                DiffusionConditionalResidualBlock1d(dim_out, dim_out, cond_dim, kernel_size, n_groups, use_film_scale),
                # Upsample as long as it is not the last block.
                nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)
        global_feature = torch.cat([timesteps_embed, global_cond], dim=-1)

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -scale)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.block(x)
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int, n_groups: int, use_film_scale: bool):
        super().__init__()
        self.use_film_scale = use_film_scale
        self.out_channels = out_channels
        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out: Tensor = self.conv1(x)

        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale:
            scale = cond_embed[:, :self.out_channels]
            bias = cond_embed[:, self.out_channels:]
            out = scale * out + bias
        else:
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


def create_stats_buffers(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: DatasetStats | None = None,
) -> nn.ModuleDict:
    stats_buffers = {}

    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        assert isinstance(norm_mode, NormalizationMode)
        if norm_mode is NormalizationMode.IDENTITY:
            continue

        shape = tuple(ft.shape)
        if ft.type is FeatureType.VISUAL:
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            shape = (c, 1, 1)  # override image shape to be invariant to height and width

        buffer = {}
        if norm_mode is NormalizationMode.MEAN_STD:
            buffer = {
                "mean": torch.tensor(stats[key]["mean"]).float() if stats else torch.full(shape, torch.nan),
                "std": torch.tensor(stats[key]["std"]).float() if stats else torch.full(shape, torch.nan),
            }
        elif norm_mode is NormalizationMode.MIN_MAX:
            buffer = {
                "min": torch.as_tensor(stats[key]["min"]).float() if stats else torch.full(shape, torch.nan),
                "max": torch.as_tensor(stats[key]["max"]).float() if stats else torch.full(shape, torch.nan),
            }

        stats_buffers[key.replace('.', '_')] = nn.ParameterDict({k: nn.Parameter(v, requires_grad=False) for k, v in buffer.items()})

    return nn.ModuleDict(stats_buffers)


class Normalize(nn.Module):
    def __init__(self, features: dict[str, PolicyFeature], norm_map: dict[str, NormalizationMode], stats: DatasetStats | None):
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats_buffers = create_stats_buffers(features, norm_map, stats)

    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        for key, ft in self.features.items():
            assert key in batch
            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            buffer = self.stats_buffers[key.replace('.', '_')]
            if norm_mode is NormalizationMode.MEAN_STD:
                batch[key] = (batch[key] - buffer["mean"]) / (buffer["std"] + 1e-8)
            elif norm_mode is NormalizationMode.MIN_MAX:
                batch[key] = (batch[key] - buffer["min"]) / (buffer["max"] - buffer["min"] + 1e-8)
                batch[key] = batch[key] * 2 - 1  # normalize to [-1, 1]

        return batch


class Unnormalize(nn.Module):
    def __init__(self, features: dict[str, PolicyFeature], norm_map: dict[str, NormalizationMode], stats: DatasetStats | None):
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats_buffers = create_stats_buffers(features, norm_map, stats)

    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, ft in self.features.items():
            assert key in batch
            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            buffer = self.stats_buffers[key.replace('.', '_')]
            if norm_mode is NormalizationMode.MEAN_STD:
                batch[key] = batch[key] * buffer["std"] + buffer["mean"]
            elif norm_mode is NormalizationMode.MIN_MAX:
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (buffer["max"] - buffer["min"]) + buffer["min"]

        return batch


if __name__ == "__main__":
    config = DiffusionConfig()
    model = DiffusionPolicy(config)
    device = torch.device(model.config.device)

    # Create dummy batch
    batch = {
        "observation.image": torch.randn(2, 2, 3, 96, 96).to(device),
        "observation.state": torch.randn(2, 2, 2).to(device),
        "action": torch.randn(2, 16, 2).to(device),
        "action_is_pad": torch.zeros(2, 16, dtype=torch.bool).to(device),
    }

    # Forward pass
    loss, _ = model.forward(batch)
    print(f"Loss: {loss.item()}")
