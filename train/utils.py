from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import math

def train_one_epoch(epoch_index,tb_writer):
    running_loss = 0.
    last_loss=0.

    for i, data in tqdm(enumerate(training_loader),desc='Batch'):
        # Every data instance is an input (X) + label (y) pair
        inputs,labels = data
        
        # zero gradients for every batch
        optimizer.zero_grad()
        # make predictions for batch
        output = model(inputs)
        
        # compute loss and gradients   
        loss = loss_fn(output,labels)
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()
            
        #Gather data and report
        running_loss += loss.item()
        if i % 10==9:
            last_loss = running_loss / 10 # avg per batch loss for last 10 batches
            print(' batch {} loss: {}'.format(i+1,last_loss))
            # to tensor board
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train',last_loss,tb_x)
            running_loss = 0.
                
    return last_loss

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