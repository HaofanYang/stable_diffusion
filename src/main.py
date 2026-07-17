# Playground for manual testing
from dataset import Dataset
from torch.utils.data import DataLoader
import utils
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    data = Dataset(root_path="./data/faces")
    sample = data[0]
    utils.show_image(sample)
    
