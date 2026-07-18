# 训练 U-Net 去预测噪声：这是把 dataset / ForwardProcess / UNet 串起来的训练循环
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import Dataset
from gaussion_process import ForwardProcess, NoiseScheduler
from unet import UNet

SCRIPT_DIR = Path(__file__).resolve().parent


def train(
    epochs=10,
    batch_size=32,
    lr=2e-4,
    image_size=(64, 64),
    timesteps=1000,
    device=None,
    checkpoint_dir=None,
    save_every=200,  # 每隔多少步存一次 checkpoint
    resume_from=None,  # 传一个 checkpoint 路径，从这个权重继续训练
):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else SCRIPT_DIR.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(root_path=SCRIPT_DIR.parent / "data" / "faces", image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    forward_process = ForwardProcess(NoiseScheduler(timesteps), steps=timesteps).to(device)
    model = UNet(in_channels=3, base_channels=64, channel_mults=(1, 2, 4)).to(device)
    if resume_from is not None:
        model.load_state_dict(torch.load(resume_from, map_location=device))
        print(f"resumed weights from {resume_from}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    step = 0
    for epoch in range(epochs):
        for x0 in dataloader:
            x0 = x0.to(device)
            t = torch.randint(0, timesteps, (x0.shape[0],), device=device)

            x_t, noise = forward_process(x0, t)
            predicted_noise = model(x_t, t)
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if step % 10 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            if step % save_every == 0 and step > 0:
                torch.save(model.state_dict(), checkpoint_dir / f"unet_step{step}.pt")
            step += 1

    torch.save(model.state_dict(), checkpoint_dir / "unet_final.pt")
    return model, losses


if __name__ == "__main__":
    train(resume_from=SCRIPT_DIR.parent / "checkpoints" / "unet_final.pt")
