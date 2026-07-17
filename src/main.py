# Playground for manual testing
from dataset import Dataset
from torch.utils.data import DataLoader
from gaussion_process import ForwardProcess
from pathlib import Path
import torch
SCRIPT_DIR = Path(__file__).resolve().parent

def visualize_diffusion_process(dataset):
    import random
    dataset_size = len(dataset)
    example = dataset[random.randint(0, dataset_size)]
    model = ForwardProcess(visualize=True)
    model.eval()
    with torch.no_grad():
        model(example)


if __name__ == "__main__":
    dataset = Dataset(root_path="./data/faces")
    visualize_diffusion_process(dataset)
