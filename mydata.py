import os
from torch.utils.data import Dataset
from PIL import Image


class MyDataset(Dataset):
    '''Dataset for concept image loading'''
    def __init__(self, base_path, transform=None):
        self.base_path = base_path
        self.transform = transform
        self.names_list = []
        for root, dirs, files in os.walk(base_path):
            self.names_list = files
        self.size = len(self.names_list)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = os.path.join(self.base_path, self.names_list[idx])
        filename = self.names_list[idx]
        if not os.path.isfile(image_path):
            print(image_path + ' does not exist!')
            return None
        with open(image_path, 'rb') as f:
            with Image.open(f) as image:
                image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return (image, filename)
