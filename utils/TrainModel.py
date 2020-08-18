# essential resources
import torch
from torch import nn
from torchvision import models

# for random_test function
import numpy as np
from utils.LIPDataset import untransform_n_display

def random_test(dataset, model, norm_param: list, display: bool = True):
    ''' Chose a random data point, perform inference. If set the 'display' flag display the results. 
    Args:
        dataset: image & keypoint dataset
        model: NN
        display(Optional): If True display the result, else no.
        norm_param: Contain mean and standard deviation of the normalized transformation
    Return: 
        the test data point and prediction
        sample: Custom datatype contain image and keypts
        pred_pts: Keypoints predicted by model
    '''
    # selecting random data element
    idx = np.random.randint(len(dataset))
    sample = dataset[idx]
    # to gpu for faster inference
    model = model.cuda()
    # inference
    image, actual_pts, pred_pts = inference(model, sample.copy())
    # transfer to cpu
    model = model.cpu()
    torch.cuda.empty_cache()

    # display the predicted
    if display:
        untransform_n_display([{'image':image, 'keypoints':pred_pts}], 0, *norm_param)

    return sample, pred_pts

def inference(model, sample):
    '''
        Perform single inference on image. Reformat input image and output predicted keypoints
    Args:
        model: NN, sample: custom dataset element (for representing human pose keypoints)
    Return:
        output_image: input image (tensor)
        actual_keypts: the ground-truth for keypoints (shape:[32]) transp
        pred_keypts: the predicted keypoints (shape:[32])
        (every element in cpu device)
    '''

    # model = model.cuda()

    image = sample['image'].unsqueeze(0)
    # inference
    model.eval()

    with torch.no_grad(): # not storing previous computational graph    
        pred = model.forward(image.cuda())

    # model = model.cpu()
    pred = pred.cpu().reshape(32)
    # torch.cuda.empty_cache()
    return sample['image'], sample['keypoints'], pred


def initialize_model( num_classes: int, use_pretrained: bool = True):
    '''
    Initialize a model from pytorch's model_zoo. Update the final layer to match the output 
    
    Args:
        num_classes: the output of final layer
        use_pretrained: If True, will download pretrained model
    '''

    model = models.wide_resnet50_2(pretrained=use_pretrained)

    set_requires_grad(model, False) # using TL as feature extractor

    num_ftrs = model.fc.in_features # input features size to fully connected layer
    mid_ftrs = num_classes + (num_ftrs - num_classes)//2 # define size of middle layer's feature size in FC layers
    # custom FC layer for every type
    fc_layer = nn.Sequential(
                    nn.Linear(num_ftrs, mid_ftrs),
                    nn.SELU(),
                    nn.Dropout(0.3),
                    nn.Linear(mid_ftrs, num_classes)
                )
    
    model.fc = fc_layer

    return model


def set_requires_grad(model, value: bool = False):
    ''' 
        Set requires_grad attributes of parameters of model. 
        If 'True' autograde will keep track of chances in value and the weights will be updated during training.
        Else, no updates in weights.
    '''
    for param in model.parameters():
        param.set_requires_grad = value
