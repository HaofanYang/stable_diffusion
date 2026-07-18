import matplotlib.pyplot as plt
import torch
from PIL import Image

def show_image(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.min() < 0:
            # 数据集用 Normalize((0.5,)*3, (0.5,)*3) 把图片变到大致 [-1, 1]，
            # 这里反归一化回 [0, 1] 再显示；加噪声后可能超出 [-1, 1]，一并 clamp 掉。
            image = (image.clamp(-1, 1) + 1) / 2
        image = image.permute(1, 2, 0).numpy()
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")
    plt.imshow(image)
    plt.axis('off')
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

def plot_training_progress(batch_idx, loss, accuracy, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(batch_idx, loss)
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax2.plot(batch_idx, accuracy)
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Accuracy')
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    else:
        plt.show(block=False)
        plt.pause(0.001)
    plt.close(fig)