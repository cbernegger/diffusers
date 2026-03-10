import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# Ziel: LJ1-01 Patch
target_channels = 1

# Bedingungen: Beispielhaft
# bm, gaia_year, urban_mask, building_height, water_mask, ndvi, ndbi, mndwi
cond_channels = 8

H = W = 128
batch_size = 2

model = UNet2DModel(
    sample_size=H,
    in_channels=target_channels + cond_channels,  # noisy target + conditions
    out_channels=target_channels,                 # predict noise for target
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 256),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
).to(device)

scheduler = DDPMScheduler(num_train_timesteps=1000)

# Dummy target (später LJ1-01)
target = torch.randn(batch_size, 1, H, W, device=device)

# Dummy conditions (später Black Marble + Geo-Layer)
cond = torch.randn(batch_size, cond_channels, H, W, device=device)

# Standard diffusion training step
noise = torch.randn_like(target)
timesteps = torch.randint(
    0, scheduler.config.num_train_timesteps, (batch_size,), device=device
).long()

noisy_target = scheduler.add_noise(target, noise, timesteps)
model_input = torch.cat([noisy_target, cond], dim=1)

noise_pred = model(model_input, timesteps).sample
loss = F.mse_loss(noise_pred, noise)

print("model_input shape:", model_input.shape)
print("noise_pred shape:", noise_pred.shape)
print("loss:", loss.item())