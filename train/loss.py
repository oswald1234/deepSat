import torch
import torch.nn as nn
import torch.nn.functional as F

""" 
# inputs  = model output, tensor(B,C,H,W), B = Batch Size, C = Classes, H = Height, W = Width
# targets = labels, tensor(B,H,W), B = Batch Size, H = Height, W = Width
# alpha   = Scalar (>= 0)
# beta    = Scalar (>= 0)
# gamma   = Scalar [1,3]
# smooth  = Smoothing factor, scalar (have seen 1e-6 and 1 used as values)

# alfa = beta = 1   = Tanimoto coefficient
# alfa = beta = 0.5 = Sørensen–Dice coefficient

# Resources:
# https://en.wikipedia.org/wiki/Tversky_index 
# https://github.com/nabsabraham/focal-tversky-unet/issues/3#issuecomment-48525
# https://www.kaggle.com/code/youssefelkilany/jaws-segmentation/notebook
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Tversky-Loss
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

""" 

# TVERSKY FOCAL LOSS MULTICLASS

class focalTverskyLoss(nn.Module):
    def __init__(self,smooth=1e-6,alpha=0.5,beta=0.5,gamma=0.5,ignore_index=[0,21,22]):
        super(focalTverskyLoss, self).__init__()
        self.ignore_index=ignore_index
        self.GAMMA = gamma
        self.BETA=beta
        self.ALPHA=alpha
        self.smooth=smooth

    def forward(self, inputs, targets):
        inputs=inputs.flatten(start_dim=-2)
        targets=targets.flatten(start_dim=-2)
        NUM_CLASSES  = inputs.shape[1]
        targets = F.one_hot(targets,num_classes=NUM_CLASSES)

        if self.ignore_index: 
            indices=torch.tensor(np.delete(np.arange(0,NUM_CLASSES),self.ignore_index ))
            inputs = inputs.index_select(dim=1,index=indices)
            targets = targets.index_select(dim=2,index=indices)
 
        # Comment out if your model contains a sigmoid or equivalent activation layer (softmax for multiclass)
        inputs = F.softmax(inputs,dim=1)       
        
        # Transpose and flatten 
        inputs=inputs.transpose(2,1)
        inputs = inputs.flatten(end_dim=-2)
        targets = targets.flatten(end_dim=-2)
     
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum(dim=0)
        FP = ((1-targets) * inputs).sum(dim=0)
        FN = (targets * (1-inputs)).sum(dim=0)
        
        # Tversky Index
        # https://en.wikipedia.org/wiki/Tversky_index 
        Tversky = (TP + self.smooth) / (TP + self.ALPHA * FN + self.BETA * FP + self.smooth)
     
        # Focal Tversky Loss
        # https://towardsdatascience.com/dealing-with-class-imbalanced-image-datasets-1cbd17de76b5
        FocalTversky = ((1 - Tversky)**self.GAMMA).sum()
  
        return FocalTversky