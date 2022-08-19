from torch.utils.data import Dataset
import h5py
import os
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

class TaxiDataset(Dataset):

    def __init__(self, root_dir="./data"):
        with h5py.File(os.path.join(root_dir, "SK_DownsampledGANFocusAreaData.h5"), 'r') as f:
            y = f.get('X_train')
            y = np.array(y, dtype=np.float32)
            images = f.get('y_train')
            images = np.array(images, dtype=np.float32)

        std1, std2 = np.std(y[:,0]), np.std(y[:, 1])
        y[:, 0] /= std1
        y[:, 1] /= std2
        print(std1, std2)
        images = np.expand_dims(images, 1)
        self.images = images
        self.y = y[:, :2]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.images[idx]*2.0 - 1.0
        y = self.y[idx]
        #image = self.transform(image)
        return image, y

if __name__ == "__main__":
    import torch
    import numpy as np
    dataset = TaxiDataset()
    print(np.max(dataset[:][0]), np.min(dataset[:][0]))
