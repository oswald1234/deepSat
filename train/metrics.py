
from sklearn.metrics import multilabel_confusion_matrix
from dataset.utils import classCount
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import math
import warnings

 # NB! See classDict.py!
strlist = [{}]*27
strlist[0] = "Unclassified"
strlist[1] = "Continuous Urban Fabric"
strlist[2] = "Discontinuous Dense Urban Fabric"
strlist[3] = "Discontinuous Medium Density Urban Fabric"
strlist[4] = "Discontinuous Low Density Urban Fabric"
strlist[5] = "Discontinuous Very Low Density Urban Fabric"
strlist[6] = "Isolated Structures"
strlist[7] = "Industrial, commercial, public, military and private units"
strlist[8] = "Roads"
strlist[9] = "Railways and associated land"
strlist[10] = "Port areas"
strlist[11] = "Airports"
strlist[12] = "Mineral extraction and dump sites"
strlist[13] = "Construction sites"
strlist[14] = "Land without current use"
strlist[15] = "Green urban areas"
strlist[16] = "Sports and leisure facilities"
strlist[17] = "Arable land (annual crops)"
strlist[18] = "Permanent crops (vineyards, fruit trees, olive groves)"
strlist[19] = "Pastures"
strlist[20] = "Complex and mixed cultivation patterns"
strlist[21] = "Orchards at the fringe of urban classes"
strlist[22] = "Forests"
strlist[23] = "Herbaceous vegetation associations (natural grassland, moors...)"
strlist[24] = "Open spaces with little or no vegetations (beaches, dunes, bare rocks, glaciers)"
strlist[25] = "Wetland"
strlist[26] = "Water bodies"

############## HOW TO COMPUTE MODEL METRICS #############
#    1. computeConfMats()
#    2. computeMetrics() (use returned metrics in step 3)
#    3. wma()
######################################################### 

# yTrue = tensor(B,H,W), B = Batch Size, H = Height, W = Width
# yPred = tensor(B,H,W), B = Batch Size, H = Height, W = Width
# Returns ndarray of shape (n_classes, 2, 2) with confusion matrices per class
def computeConfMats(yTrue,yPred):
    LABELS = np.arange(27) # (0,1,2...,26)
    # Flatten dimensions BxHxW --> B*H*W
    yTrue = yTrue.reshape(-1)
    yPred = yPred.reshape(-1)
    cMats = multilabel_confusion_matrix(y_true=yTrue,y_pred=yPred,labels=LABELS)
    return cMats

# Accuracy
def acc(cMat):
    tn, fp, fn, tp = cMat.ravel()
    metric = (tp+tn)/cMat.sum()
    return metric
       
# Precision
def prc(cMat):
    tn, fp, fn, tp = cMat.ravel()
    if (tp+fp) != 0:
        metric = tp/(tp+fp)
    else:
        metric = 0.0
        warnings.warn("Precision: Division by zero. Metric is undefined and set to 0")
    return metric
    
# Recall
def rcl(cMat):
    tn, fp, fn, tp = cMat.ravel()
    if (tp+fn) != 0:
        metric = tp/(tp+fn)
    else:
        metric = 0.0
        warnings.warn("Recall: Division by zero. Metric is undefined and set to 0")
    return metric

# F1-Score
def f1s(cMat):
    tn, fp, fn, tp = cMat.ravel()
    if (fp+tp+tp+fn) != 0:
        metric = (2*tp)/(fp+tp+tp+fn)
    else:
        metric = 0.0
        warnings.warn("F1-Score: Division by zero. Metric is undefined and set to 0")
    return metric

# Intersection over Union
def iou(cMat):
    tn, fp, fn, tp = cMat.ravel()
    if (fp+tp+fn) != 0:
        metric = tp/(fp+tp+fn)
    else:
        metric = 0.0
        warnings.warn("IoU: Division by zero. Metric is undefined and set to 0")
    return metric
    
# Matthews Correlation Coefficient
def mcc(cMat):
    tn, fp, fn, tp = cMat.ravel()
    numerator = tp*tn-fp*fn
    denominator = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    if denominator != 0:
        metric = numerator/math.sqrt(denominator)
    else:
        metric = 0.0
        warnings.warn("MCC: Division by zero. Metric is undefined and set to 0")
    return metric

# cMats = ndarray of shape (n_classes, 2, 2)
# Returns computed metrics per class, ndarray of shape (n_classes, n_metrics)
def computeMetrics(cMats):
    metrics=torch.empty(27,6,dtype=torch.float)
    for i, cMat in enumerate (cMats):
        metrics[i][0] = acc(cMat)
        metrics[i][1] = prc(cMat)
        metrics[i][2] = rcl(cMat)
        metrics[i][3] = f1s(cMat)
        metrics[i][4] = iou(cMat)
        metrics[i][5] = mcc(cMat)     
    return metrics

# Weighted Macro-Average model metrics
# metrics per class       = ndarray of shape (n_classes, n_metrics)
# training_loader         = DataLoader(dataset)
# c                       = 27 (number of classes including unclassified class)
# Returns weighted macro-averaged model metrics
def wma(metrics,training_loader,c=27):
    trainiter = iter(training_loader)
    classCounts,_ = classCount(trainiter,c)
    metricsWMA = torch.zeros(len(metrics[1]),dtype=torch.float)
    weights = torch.zeros(len(classCounts),dtype=torch.float)
    weights = classCounts/classCounts.sum()
    
    # Permute metrics (n_classes, n_metrics) --> (n_metrics, n_classes)
    metrics = torch.permute(metrics,(1,0))
    
    # Multiply each class metric with class weight and sum to get
    # weighted macro-average model metric
    for i, metric in enumerate(metrics):
        metricsWMA[i] = np.multiply(weights,metric).sum()
    
    return metricsWMA # (Accuracy,Precision,Recall,F1-Score,IoU,MCC)

############################ Metric Visualization ##########################

# Prints table with metrics per class
# metrics = = ndarray of shape (n_classes, n_metrics)
def printClassMetrics(metrics):
    d = {}
    for i in range(len(metrics)):
        d[i] = [metrics[i][0].item(), metrics[i][1].item(), metrics[i][2].item(), 
                metrics[i][3].item(), metrics[i][4].item(), metrics[i][5].item(),strlist[i]]

    print("@@ Class Metrics @@\n")

    print ("{:<8} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        'Label','Accuracy','Precision','Recall','F1-Score','IoU','MCC','Description'))

    for l, v in d.items():
        acc, prc, rcl, f1s, iou, mcc, dsc = v
        print ("{:<8} {:<15.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10}".format(
            l, acc, prc, rcl, f1s, iou, mcc, dsc))

# Prints weighted macro-averaged model metrics 
# metrics = ndarray with model metrics of shape (n_metrics)
def printMetrics(metrics):  
    strlistM = [{}]*len(metrics)
    strlistM[0] = "Accuracy"
    strlistM[1] = "Precision"
    strlistM[2] = "Recall"
    strlistM[3] = "F1-Score"
    strlistM[4] = "Intersection over Union"
    strlistM[5] = "Matthews Correlation Coefficient"

    print("@@ Weighted Macro-Average Model Metrics @@\n")
    
    for i in range(len(metrics)):
        print(f'{strlistM[i]:<35}: {metrics[i].item():.4f}')

# Prints confusion matrices per class        
# cMats = ndarray of shape (n_classes, 2, 2) with confusion matrices per class
def printConfusionMatrices(cMats): 
    labels = ["0","1","2","3","4","5","6","7","8","9","10",
              "11","12","13","14","15","16","17","18","19",
              "20","21","22","23","24","25","26"]

    fig, axes = plt.subplots(nrows=9, ncols=3, figsize=(20,45))

    group_names = ["True Negative","False Positive","False Negative","True Positive"]

    counter = 0

    for row in axes:
        for col in row:
            group_counts = ["{0:0.0f}".format(value) for value in cMats[counter].flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in cMats[counter].flatten()/np.sum(cMats[counter])]
            labels2 = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
            labels2 = np.asarray(labels2).reshape(2,2)

            sns.set(font_scale=1.1)
            sns.heatmap(cMats[counter], annot=labels2, fmt="", cmap='Blues', ax=col)
            col.set_title("Label " + labels[counter] + ": " + strlist[counter], fontsize=12)
            col.set_xlabel("Predicted Label", fontsize=12)
            col.set_ylabel("True Label", fontsize=12)

            counter+=1

    plt.tight_layout()  
    plt.show()