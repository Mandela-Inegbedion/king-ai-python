# dataset.py
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

class TestDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.files = os.listdir(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

test_dataset = TestDataset("path/to/test/images", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
