# essential resources
import torch
from torch import nn
from torchvision import models

# for random_test function
import numpy as np
from utils.LIPDataset import untransform_n_display

class TrainModel(object):
    ''' A class for performing training loops over datasets '''
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer):
        '''
            Args:
                model: NN
                train_loader, valid_loader: Data loaders
                criterion: Loss function (Just class, not instance)
                optimizer: Model weight optimizer (Just class)
        '''
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion() # creating instance here
        self.optimizer_cls = optimizer
        self.lr = 0.0 # dummy value

    def __call__(self, epoch: int, lr: float = 0.0003, eval_freq: int = 20, save_model: bool = False):
        '''
        Perform forward, backward pass and weight update. Evaluate performance on 
        validation data based on value of value of variabel 'eval_freq' per epoch.
        Optionally save model based on condition of decrease in model loss.
        Args: 
            epoch: Number of training loop over whole training data
            lr: Learning rate for training
            eval_freq: Frequency of performing and printing - train and validation loss per epoch
            save_model: If True and valid loss decrease, save model stats after every evaluation of performance on validation data 
        Return: 
            train_loss (list): Contain record of values of loss on training data
            valid_loss (list): Contain record of values of loss on validation data
        '''
        assert(isinstance(epoch, int), 'The variable should be an int datatype')
        # OPTIMIZER
        if self.lr != lr: # new learning rate
            self.lr = lr
            # initialize new optimizer (since lr is changed)
            self.optimizer = self.optimizer_cls(self.model.fc.parameters(), lr=self.lr)

        # storing loss over whole training
        train_loss = list()
        valid_loss = list()
        # perfrom evaluation and print results ''eval_freq' times per epoch
        num_batches = len(self.train_loader)
        prt_idx = num_batches // eval_freq

        # prepare the net for training (autograd, dropout -> on)
        self.model = self.model.cuda()
        self.model.train()

        for i in range(epoch): # looping over whole dataset
            running_loss = 0.0
            # training on batches of data
            for batch_i, samples in enumerate(self.train_loader):
                # get images and keypoints
                images, keypts = samples['image'].cuda(), samples['keypoints'].cuda()
                # forward pass
                pred = self.model.forward(images)
                # compute loss
                loss = self.criterion(pred, keypts)
                # zero the accumulate weight gradients
                self.optimizer.zero_grad()
                # backward pass (calculate current weight gradients)
                loss.backward()
                # update the weights
                self.optimizer.step()

                running_loss += loss.item() # storing loss

                # Checking performance on validation data
                # Print all the losses (frequency  is based on prt_freq value)
                if batch_i % prt_idx == 0: 

                    temp_loss = 0.0
                    with torch.no_grad(): # switching off autograd engine, save memory and computation
                        self.model.eval() # switching off dropout and batchnorm  
                        for samples in self.valid_loader: # looping over batches
                            # get images and keypoints
                            images, keypts = samples['image'].cuda(), samples['keypoints'].cuda()
                            # forward pass
                            pred = self.model.forward(images)
                            # compute loss
                            loss = self.criterion(pred, keypts)
                            
                            temp_loss += loss.item()

                    avg_train_loss = running_loss/prt_idx
                    avg_valid_loss = temp_loss/len(self.valid_loader)
                    print(f'Epoch: {i}, Batch: {batch_i}, Train Avg. Loss: {avg_train_loss}, Validation Avg. Loss:{avg_valid_loss}')
                    # storing avg loss for analysis later
                    train_loss.append(avg_train_loss)
                    valid_loss.append(avg_valid_loss)
                    running_loss = 0.0
                    self.model.train() # return to trian mode for next loop

                    break # break after one turn

        print("Training complete")
        # transfer to cpu
        self.model = self.model.cpu()
        torch.cuda.empty_cache()

        return train_loss, valid_loss

        


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


