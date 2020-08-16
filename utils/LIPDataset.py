# import required packages
from torch.utils.data import Dataset
import pandas as pd
import os
import cv2
# for transforms
import torch
from torchvision import transforms
import torch.nn.functional as F
import numpy as np

def untransform_n_display(dataset, index : int, mean, std):
    ''' Transform  the sample to display (Mainly converting from tensor to numpy array)
    Args:
        dataset : Keypoints dataset 
        index : index of datapoint to use
        mean, std (sequence): Used to normalize image

    '''
    sample = dataset[index] # extract sample at index
    image = sample['image'] # extract image

    sample['keypoints'] = sample['keypoints'].data.numpy()
    sample['image'] = sample['image'].data.numpy() # data in image and current form of matrix
    sample = unNormalize(sample, mean, std) # unNormalize
    sample['image'] = sample['image'].transpose((1,2,0)).astype(np.uint8) # change dtype to correct format for display


    plot_data(sample)

def unNormalize(sample, mean, std):
    '''
        Undo the normalization using mean and std for image.
        For keypoints, standard figures are used
        image (ndarray, type=Float)
    '''


    # IMAGE OPERATIONS  
    image = sample['image']
    # from (approx) [-1,1] to [0,1]
    for i in range(3):
        image[i] = (image[i] * std[i] + mean[i])
    sample['image'] = image*255.0 # [0, 1] to [0, 255]
    
    # KEYPOINT OPERATIONS  
    keypts = sample['keypoints']
    keypts = keypts * 0.5 + 0.5 # from [-1, 1] -> [0, 1] 
    max_keypts = sample['image'].shape[1] # maximum possible value for keypoints
    keypts *= max_keypts # changing value range from [0, 1] -> [0, max_keypts] 
    sample['keypoints'] = keypts

    return sample
    

class Normalize(transforms.Normalize):
    '''
        Normalize the values of sample (tensor image and keypoints)
        This class inherit torchvision.transforms.Normalize class to perfrom normalization on tensor image.
    '''
    
    def __init__(self, mean, std, inplace: bool = False):
        '''
        Args:
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channel.
            inplace : Bool to make this operation in-place.
        '''
        super().__init__(mean, std, inplace)

    def __call__(self, sample:dict):
        """
        Args:
            sample : Contain image (3-dimensional) and keypoints

        Returns:
            Normalized image and keypoints
        """
        img = sample['image'] / 255.0 # changing value range from [0, 255] -> [0, 1]
        sample['image'] = super().__call__(img) # normalize image

        max_keypts = sample['image'].shape[1] # maximum possible value for keypoints
        keypts = sample['keypoints']
        keypts /= max_keypts # changing value range from [0, max_keypts] -> [0, 1]
        sample['keypoints'] = (keypts - 0.5) / 0.5 # from [0, 1] -> [-1, 1]

        return sample

class RandomHorizontalFlip(object):
    """ 
        Horizontally flip the given image with a given probability.
        Keypoints are changed accordingly. It takes only ndarray images and not tensor datatype
    """

    def __init__(self, p=0.5):
        """
        Args:
            p : the probability of sample being flipped. Default value is 0.5
        """
        self.p = p

    def __call__(self, sample:dict):
        """
        Args:
            sample : Contain image and keypoints

        Returns:
            Randomly flipped image and keypoints
        """
        if np.random.rand(1) < self.p:
            image, key_pts = sample['image'], sample['keypoints']

            w, h = image.shape[:2]
            key_pts = np.asarray([w - key_pts[i]  if i%2==0 and key_pts[i]>0 else key_pts[i] for i in range(len(key_pts))])
            return {'image':cv2.flip(image, 1), 'keypoints':key_pts}

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
         
        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)
            
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        key_pts = np.asarray(key_pts, dtype=np.float)
        
        return {'image': torch.from_numpy(image).type(torch.FloatTensor),
                'keypoints': torch.from_numpy(key_pts).type(torch.FloatTensor)}
class RandomCrop(object):
    """
        Randomly crop the image in a sample. Accordingly, the range of keypoints will be changed.
    """

    def __init__(self, output_size:[int, tuple, list]):
        '''
        Args:
            output_size : Desired output size. If int, square crop is made.
            If tuple or list, output is matched exactly.
        '''
        assert isinstance(output_size, (int, tuple, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        # resize keypoints
        key_pts = [key_pts[i] - (left * (1 - i%2) + top * (i%2))  for i in range(len(key_pts))]

        return {'image':image, 'keypoints':key_pts}

    
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

        image = cv2.resize(image, (new_w, new_h))

        # resize keypoints
        key_pts = [key_pts[i] * (new_w/w * (1 - i%2) + new_h/h * (i%2))  for i in range(len(key_pts))]

        return {'image':image, 'keypoints':key_pts}

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