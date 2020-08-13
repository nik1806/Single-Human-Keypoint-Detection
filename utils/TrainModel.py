# essential resources
from torch import nn
from torchvision import models

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
