from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import math
import torch

# NB! When calling on confMat() and metric functions, save all individual class
# confusion matrices and metrics. The data will be used as input arguments when
# calling on weightsWMA() and wma() to compute a weighted metric.

# Computes confusion matrix.
# yTrue = True class (array)
# yPred = Predicted class (array)
def confMat(yTrue,yPred):
    cMat = confusion_matrix(yTrue,yPred)
    return cMat

# cMat = Confusion matrix for a single class

# Accuracy
def acc(cMat):
    tn, fp, fn, tp = cMat.ravel()
    metric = (tp+tn)/cMat.sum()
    return metric
       
# Precision
def prc(cMat):
    tn, fp, fn, tp = cMat.ravel()
    metric = tp/(tp+fp)
    return metric
    
# Recall
def rlc(cMat):
    tn, fp, fn, tp = cMat.ravel()
    metric = tp/(tp+fn)
    return metric

# F1-Score
def f1s(cMat):
    tn, fp, fn, tp = cMat.ravel()
    metric = (2*tp)/(fp+tp+tp+fn)
    return metric

# Intersection over Union
def iou(cMat):
    tn, fp, fn, tp = cMat.ravel()
    metric = tp/(fp+tp+fn)
    return metric
    
# Matthews Correlation Coefficient
def mcc(cMat):
    tn, fp, fn, tp = cMat.ravel()
    numerator = tp*tn-fp*fn
    denominator = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    metric = numerator/math.sqrt(denominator)
    return metric
    
# Computes class weights for wma().
# cMats = Confusion matrices for all classes (array)
def weightsWMA(cMats):
    nClass = []
    nAll = 0
    weights = []  
    for cMat in cMats:
        nClass.append(cMat.sum())
        nAll = nAll+cMat.sum()  
    for c in nClass:
        weights.append(c/nAll)
    return weights
 
# Weighted Macro-Average
    # Call on weightsWMA() before calling this function
    # metricVals = Metric X values for all classes (array)
    # weights = WMA class weights (array)
def wma(metricVals,weights):
    metricWMA=np.multiply(weights,metricVals).sum()
    return metricWMA

# Computes min (2 %) & max (98 %) percentile value along column (0) and rows (1)
# x = Image band (array)
def minMaxPercentile(x,per=2):
	minPer = np.percentile(x,per,axis=(0,1))
	maxPer = np.percentile(x,100-per,axis=(0,1))
	return minPer,maxPer

# trainiter = iter(DataLoader(dataset))
# c         = 27 (number of classes including unclassified class)
def weights(trainiter,c=27):
    # Count all classes/labels in the enitre dataset and per sample
    # Compute weights wrt the entire dataset and per sample
    
    n_classes_dataset = torch.zeros(c,dtype=torch.int)
    class_weights_dataset = torch.zeros(c,dtype=torch.float)
    
    n_classes_sample = torch.zeros(len(trainiter),c,dtype=torch.int)
    class_weights_sample = torch.zeros(len(trainiter),c,dtype=torch.float)
    
    # Count classes/labels in entire dataset. Save total count and count per sample
    # i = samples/patches in dataset, j=examples/instances/rows in sample/patch
    for i, (_, labels) in enumerate(trainiter):
        classes,count = labels.unique(return_counts=True)
        for j in classes:
            idx = (classes==j).nonzero().item()
            classCount = count[idx].item()
            n_classes_dataset[j] += classCount
            n_classes_sample[i][j] = classCount
    
    ############### Weights wrt sample
    
    # Set class weight = 0 for label = 0
    # Inf control. Init torch.zero not actually "zero"
    for i in range(len(n_classes_sample)):
        for j in range(len(n_classes_sample[i])):
            if j == 0:
                class_weights_sample[i][j] = 0
            elif n_classes_sample[i][j].item() != 0:
                class_weights_sample[i][j] = torch.max(n_classes_sample[i])/n_classes_sample[i][j]
                
    ################ Weights wrt dataset
    
    #class_weights_dataset=n_classes_dataset/n_classes_dataset.sum()
    
    # Set class weight = 0 for label = 0
    #class_weights_dataset[0] = 0
    
    # Inf control. Init torch.zero not actually "zero"
    for i, count in enumerate(n_classes_dataset):
        if i == 0:
            class_weights_dataset[i] = 0
        elif count.item() != 0:
            class_weights_dataset[i] = torch.max(n_classes_dataset)/n_classes_dataset[i]
    
    return class_weights_dataset, class_weights_sample
            
    ################
