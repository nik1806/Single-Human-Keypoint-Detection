# import required packages
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
# for transforms
import torch
from torchvision import transforms

class Resize(object):
    """
        Rescale the images to a given size. Accordingly, the range of keypoints will be changed.    
    """

    def __init__(self, output_size:[int, tuple, list]):
        '''
        Args:
            output_size : Desired output size. If int,smaller edge of image is matched with size.
            If tuple or list, output is matched exactly.
        '''
        assert isinstance(output_size, (int, tuple, list))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]

        # resize image
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))

        # resize keypoints
        key_pts = [key_pts[i] * (new_w/w * (1 - i%2) + new_h/h * (i%2))  for i in range(len(key_pts))]

        return {'image':img, 'keypoints':key_pts}

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

def plot_data(sample):
    ''' Drawing joints and bones on images '''

    img = sample['image'] # retrive image
    rec = sample['keypoints'] # retrive points coordinates

    bombs = [[0,1],[1,2]
            ,[3,4],[4,5]
            ,[6,7],[7,8],[8,9]
            ,[10,11],[11,12]
            ,[13,14],[14,15] ]
    colors = [(255,0,0),(255,0,0),
              (0,255,0),(0,255,0),
              (0,0,255),(0,0,255),(0,0,255),
              (128,128,0),(128,128,0),
              (128,0,128),(128,0,128)]

    for b_id in range(len(bombs)):
        b = bombs[b_id]
        color = colors[b_id]
        x1 = rec[ b[0] * 2 + 0]
        y1 = rec[ b[0] * 2 + 1]

        x2 = rec[ b[1] * 2 + 0]
        y2 = rec[ b[1] * 2 + 1]

        if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
            img = cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), color, 3) 
        elif x1 > 0 and y1 > 0:
            img = cv2.circle(img, (int(x1), int(y1)), 4, color, 4) 
        elif x2 > 0 and y2 > 0:
            img = cv2.circle(img, (int(x2), int(y2)), 4, color, 4)
    
    cv2.imshow('Human keypoints', img)
    cv2.waitKey(0)
        
    cv2.destroyAllWindows()