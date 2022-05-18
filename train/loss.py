import torch
import torch.nn as nn

""" 
### TVERSKY INDEX (MULTICLASS)

# inputs  = model output, tensor(B,C,H,W), B = Batch Size, C = Classes, H = Height, W = Width
# targets = labels, tensor(B,H,W), B = Batch Size, H = Height, W = Width
# alpha   = Scalar (>= 0)
# beta    = Scalar (>= 0)
# smooth  = Smoothing factor, scalar (have seen 1e-6 and 1 used as values)

# alfa = beta = 1   = Tanimoto coefficient
# alfa = beta = 0.5 = Sørensen–Dice coefficient

def tverskyIndex(inputs, targets, smooth=1, alpha=0.7, beta=0.3):
    
    # Slice first C index? C = 0 is label 0 = Unclassified
    inputs = inputs[:,1:len(inputs[0]),:,:]
    
    NUM_CLASSES = inputs.size(dim=1)

    # Permute inputs BxCxHxW into CxBxHxW, to have classes as the frist index
    inputs = inputs.permute(1,0,2,3)
    
    # Then, flatten all other input dimensions CxBxHxW --> CxB*H*W
    # and targets dimensions BxHxW --> B*H*W
    inputs = inputs.reshape(NUM_CLASSES, -1)
    targets = targets.reshape(-1)
    
    # Permute inputs CxBHW --> BHWxC
    inputs = inputs.permute(1,0)
    
    # Filter out label = 0 (Unclassified)
    targets_fltr = targets[targets != 0]
    inputs = inputs[targets != 0]
    
    # Permute inputs back BHWxC --> CxBHW
    inputs = inputs.permute(1,0)
    
    TP = (inputs * targets_fltr).sum(dim=1)
    FP = ((1-targets_fltr) * inputs).sum(dim=1)
    FN = (targets_fltr * (1-inputs)).sum(dim=1)
    
    # https://en.wikipedia.org/wiki/Tversky_index
    # https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Tversky-Loss
    # tversky_index = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
    
    # https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
    tversky_index = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
    
    return tversky_index

### FOCAL TVERSKY LOSS (MULTICLASS)

# Resources: 
# https://github.com/nabsabraham/focal-tversky-unet/issues/3#issuecomment-48525
# https://www.kaggle.com/code/youssefelkilany/jaws-segmentation/notebook
# https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Focal-Tversky-Loss

# The Focal Tversky Loss is an easy drop in solution to deal with class imbalance. 
# Although the best parameter values will take some trial and error to determine, 
# you should see good results with the following: alpha = 0.7, beta = 0.3, gamma = 3/4

# inputs  = model output, tensor(B,C,H,W), B = Batch Size, C = Classes, H = Height, W = Width
# targets = labels, tensor(B,H,W), B = Batch Size, H = Height, W = Width
# alpha   = Scalar (>= 0)
# beta    = Scalar (>= 0)
# gamma   = Scalar [1,3]
# smooth  = Smoothing factor, scalar (have seen 1e-6 and 1 used as values)

class focalTverskyLoss(nn.Module):
    def __init__(self, smooth=1,alpha=0.7,beta=0.3, gamma=4/3):
        super(focalTverskyLoss, self).__init__()
        self.smooth=smooth
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma

    def forward(self, inputs, targets):
        
        ##### Comment out if the model contains a softmax activation layer!!!!!! #####
        inputs = torch.nn.functional.softmax(inputs, dim=1)
         
        # .sum() as in Cross Entropy Loss
        fTverskyLoss = ((1 - tverskyIndex(inputs, targets, self.smooth, self.alpha, self.beta)) ** self.gamma).sum()
        
        # fTverskyLoss = ((1 - tverskyIndex(inputs, targets, smooth, alpha, beta)) ** gamma).mean()
        
        return fTverskyLoss

 """

import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import multilabel_confusion_matrix


def tverskyIndex(preds, targets, smooth=1, alpha=0.7, beta=0.3):
  
    # make target and labels same shape
    targets = targets.reshape(-1)
    preds = preds.reshape(-1)
  
    # Filter out label = 0 (Unclassified) from target and labels
    pred = preds[targets != 0]
    targets = targets[targets != 0]
    
    print(preds.shape,targets.shape)
    
    cm=multilabel_confusion_matrix(targets.cpu(),pred.cpu())
    FP = cm[:,0,1]
    FN = cm[:,1,0]
    TP = cm[:,1,1]

    # https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
    tversky_index = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
    
    return tversky_index

### FOCAL TVERSKY LOSS (MULTICLASS)

# Resources: 
# https://github.com/nabsabraham/focal-tversky-unet/issues/3#issuecomment-48525
# https://www.kaggle.com/code/youssefelkilany/jaws-segmentation/notebook
# https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Focal-Tversky-Loss

# The Focal Tversky Loss is an easy drop in solution to deal with class imbalance. 
# Although the best parameter values will take some trial and error to determine, 
# you should see good results with the following: alpha = 0.7, beta = 0.3, gamma = 3/4

# inputs  = model output, tensor(B,C,H,W), B = Batch Size, C = Classes, H = Height, W = Width
# targets = labels, tensor(B,H,W), B = Batch Size, H = Height, W = Width
# alpha   = Scalar (>= 0)
# beta    = Scalar (>= 0)
# gamma   = Scalar [1,3]
# smooth  = Smoothing factor, scalar (have seen 1e-6 and 1 used as values)

class focalTverskyLoss(nn.Module):
    def __init__(self, smooth=1,alpha=0.7,beta=0.3, gamma=4/3,ignore_index=0):
        super(focalTverskyLoss, self).__init__()
        self.smooth=smooth
        self.alpha=alpha
        self.beta=beta
        self.gamma=gamma
        self.ignore_index=ignore_index

    def forward(self, inputs, targets):

        if self.ignore_index: 
            NUM_CLASSES  = input.shape[1]
            indices=torch.tensor(np.delete(np.arange(0,NUM_CLASSES),self.ignore_index ))
            input = input.index_select(dim=1,index=indices)
        
        inputs = torch.nn.functional.softmax(inputs, dim=1)
        preds = torch.argmax(inputs,dim=1)

        fTverskyLoss = ((1 - tverskyIndex(preds, targets, self.smooth, self.alpha, self.beta)) ** self.gamma).sum()

        return fTverskyLoss
