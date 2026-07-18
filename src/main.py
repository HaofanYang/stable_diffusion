# Playground for manual testing
from dataset import Dataset
from torch.utils.data import DataLoader
from gaussion_process import ForwardProcess
from pathlib import Path
import torch
from auto_encoder import DummyDecoder, DummyEncoder
from utils import show_image

SCRIPT_DIR = Path(__file__).resolve().parent

def visualize_diffusion_process(dataset):
    import random
    dataset_size = len(dataset)
    index = random.randint(0, dataset_size - 1)
    example = dataset[index].unsqueeze(0)
    model = ForwardProcess(visualize=True)
    model.eval()
    with torch.no_grad():
        for t_val in [0, 50, 100, 250, 500, 750, 999]:
            t = torch.tensor([t_val])
            model(example, t)

def test_image_auto_encoder(dataset):
    import random
    dataset_size = len(dataset)
    index = random.randint(0, dataset_size - 1)
    example = dataset[index].unsqueeze(0)
    encoder = DummyEncoder()
    encoder.eval()
    decoder = DummyDecoder(example.shape[1:])
    decoder.eval()
    with torch.no_grad():
        show_image(example.squeeze(0))
        img_encoded = encoder(example)
        img_decoded = decoder(img_encoded)
        show_image(img_decoded.squeeze(0))

if __name__ == "__main__":
    dataset = Dataset(root_path=SCRIPT_DIR.parent / "data" / "faces")
    visualize_diffusion_process(dataset)
    test_image_auto_encoder(dataset)
