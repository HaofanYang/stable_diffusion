import matplotlib.pyplot as plt
import torch
from PIL import Image

def show_image(image):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")
    plt.imshow(image)
    plt.axis('off')
    plt.show()

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