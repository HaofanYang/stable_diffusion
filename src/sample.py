# 从训练好的 U-Net 采样：加载 checkpoint，走 inference 公式，从纯噪声一路走回图片
from pathlib import Path
import torch
from gaussion_process import NoiseScheduler
from unet import UNet
from utils import show_image

SCRIPT_DIR = Path(__file__).resolve().parent


def load_model(checkpoint_path, device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model = UNet(in_channels=3, base_channels=64, channel_mults=(1, 2, 4)).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def sample(
    model,
    image_size=(64, 64),
    batch_size=1,
    timesteps=1000,
    device=None,
    visualize=False,
    visualize_every=50,  # 每隔多少步 show_image 一次；想每步都看就设成 1
):
    device = device or next(model.parameters()).device

    # 跟 ForwardProcess 用的是同一套 noise schedule，这里重新算一遍是为了拿到
    # betas / alphas / alphas_cumprod 这些逐步采样公式需要的量（不只是它们的 sqrt）
    noise_scheduler = NoiseScheduler(timesteps)
    betas = noise_scheduler.linear_beta_schedule().float().to(device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    x = torch.randn(batch_size, 3, *image_size, device=device)  # x_T ~ N(0, I)

    for t_val in reversed(range(timesteps)):
        t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
        predicted_noise = model(x, t)

        alpha_t = alphas[t_val]
        alpha_bar_t = alphas_cumprod[t_val]
        beta_t = betas[t_val]

        # mu_theta(x_t, t) = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * epsilon_theta)
        mean = (1 / torch.sqrt(alpha_t)) * (
            x - beta_t / torch.sqrt(1 - alpha_bar_t) * predicted_noise
        )

        if t_val > 0:
            sigma_t = torch.sqrt(beta_t)
            z = torch.randn_like(x)
            x = mean + sigma_t * z
        else:
            x = mean  # 最后一步不加噪声

        if visualize and (t_val % visualize_every == 0 or t_val == 0):
            show_image(x[0])

    return x


if __name__ == "__main__":
    checkpoint_path = SCRIPT_DIR.parent / "checkpoints" / "unet_final.pt"
    model = load_model(checkpoint_path)
    images = sample(model, batch_size=1, visualize=True)
    show_image(images[0])
