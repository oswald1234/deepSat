from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import math


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

# Computes Global Min-Max normalization using specified percentiles.
# Call on minMaxPercentile() before using this function
# x = Image band (array)
# minPer = Min percentile (value)
# maxPer = Max percentile (value) 
def normalizeData(x,minPer,maxPer):
    return (x-minPer)/(maxPer-minPer)