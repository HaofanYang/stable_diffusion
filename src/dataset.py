from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from pathlib import Path



class Dataset(Dataset):
    def __init__(
        self,
        root_path,
        transform=None,
        image_size=(64, 64)
    ):
        self.root_path = Path(root_path)
        self.paths = sorted(self.root_path.glob("*.jpg"))
        self.transform = T.transforms.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]) if transform is None else transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx])
        return self.transform(img)