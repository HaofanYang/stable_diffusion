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
    index = random.randint(0, dataset_size)
    example = dataset[index].unsqueeze(0)
    model = ForwardProcess(visualize=True)
    model.eval()
    with torch.no_grad():
        for t_val in [0, 50, 100, 250, 500, 750, 999]:
            t = torch.tensor([t_val])
            model(example, t)


if __name__ == "__main__":
    dataset = Dataset(root_path="./data/faces")
    visualize_diffusion_process(dataset)
