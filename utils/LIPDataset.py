# import required packages
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2

class LIPDataset(Dataset):
    ''' Custom dataset class to handle LIP data '''

    def __init__(self, ftr_file, root_dir, transform=None):
        """
        Args:
            ftr_file (string): Path to feather file
            root_dir (string): Path to data images
            transform (callable, optional): Optional transformations
        """
        self.key_pts = pd.read_feather(ftr_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.key_pts.iloc[idx, 0])
        img = cv2.imread(img_name, cv2.IMREAD_COLOR) # RGB format

        keypts = self.key_pts.iloc[idx, 1:].values
        sample = {'image':img, 'keypoints':keypts}

        if self.transform:
            sample = self.transform(sample)

        return sample

