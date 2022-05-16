from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import math
import warnings
import itertools

# Class descriptions. See classDict.py!
n_classes_dsc = 27
 
strlist = [{}]*n_classes_dsc 
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

# Urban Atlas 2018 IDs
n_classes_ua = 26

strlist_ua = [{}]*n_classes_ua
strlist_ua[0] = "11100"
strlist_ua[1] = "11210"
strlist_ua[2] = "11220"
strlist_ua[3] = "11230"
strlist_ua[4] = "11240"
strlist_ua[5] = "11300"
strlist_ua[6] = "12100"
strlist_ua[7] = "12210/12220"
strlist_ua[8] = "12230"
strlist_ua[9] = "12300"
strlist_ua[10] = "12400"
strlist_ua[11] = "13100"
strlist_ua[12] = "13300"
strlist_ua[13] = "13400"
strlist_ua[14] = "14100"
strlist_ua[15] = "14200"
strlist_ua[16] = "21000"
strlist_ua[17] = "22000"
strlist_ua[18] = "23000"
strlist_ua[19] = "24000"
strlist_ua[20] = "25000"
strlist_ua[21] = "31000"
strlist_ua[22] = "32000"
strlist_ua[23] = "33000"
strlist_ua[24] = "40000"
strlist_ua[25] = "50000"

############################# HOW TO COMPUTE MODEL METRICS ########################
#    1. computeConfMats()
#    2. computeMetrics() (use returned class metrics in step 4)
#    3. classCount()     (in dataset/utils.py, use returned class counts in step 4)
#    4. wma()
###################################################################################

# yTrue       = tensor(B,H,W), B = Batch Size, H = Height, W = Width
# yPred       = tensor(B,H,W), B = Batch Size, H = Height, W = Width
# n_classes   = number of classes (scalar)
# Returns ndarray of shape (n_classes, 2, 2) with confusion matrices per class
def computeConfMats(yTrue,yPred,n_classes=27):
    LABELS = np.arange(n_classes) # (0,1,2..., n_classes)
    # Flatten dimensions BxHxW --> B*H*W
    yTrue = yTrue.reshape(-1)
    yPred = yPred.reshape(-1)
    cMats = multilabel_confusion_matrix(y_true=yTrue,y_pred=yPred,labels=LABELS)
    return cMats
    
# Cast to float32 to avoid data overflow    
# tn, fp, fn, tp = tensor (scalar)
def castToFloat32(tn, fp, fn, tp):
    tn=tn.to(torch.float32)
    fp=fp.to(torch.float32)
    fn=fn.to(torch.float32)
    tp=tp.to(torch.float32)
    return tn, fp, fn, tp

# Accuracy
def acc(cMat):
     tn, fp, fn, tp = cMat.ravel()
     tn, fp, fn, tp = castToFloat32(tn, fp, fn, tp)
     metric = (tp+tn)/cMat.sum()
     return metric
       
# Precision
def prc(cMat):
    tn, fp, fn, tp = cMat.ravel()
    tn, fp, fn, tp = castToFloat32(tn, fp, fn, tp)
    if (tp+fp) != 0:
        metric = tp/(tp+fp)
    else:
        metric = 0.0
        warnings.warn("Precision: Division by zero. Metric is undefined and set to 0")
    return metric
    
# Recall
def rcl(cMat):
    tn, fp, fn, tp = cMat.ravel()
    tn, fp, fn, tp = castToFloat32(tn, fp, fn, tp)
    if (tp+fn) != 0:
        metric = tp/(tp+fn)
    else:
        metric = 0.0
        warnings.warn("Recall: Division by zero. Metric is undefined and set to 0")
    return metric

# F1-Score
def f1s(cMat):
    tn, fp, fn, tp = cMat.ravel()
    tn, fp, fn, tp = castToFloat32(tn, fp, fn, tp)
    if (fp+tp+tp+fn) != 0:
        metric = (2*tp)/(fp+tp+tp+fn)
    else:
        metric = 0.0
        warnings.warn("F1-Score: Division by zero. Metric is undefined and set to 0")
    return metric

# Intersection over Union
def iou(cMat):
    tn, fp, fn, tp = cMat.ravel()
    tn, fp, fn, tp = castToFloat32(tn, fp, fn, tp)
    if (fp+tp+fn) != 0:
        metric = tp/(fp+tp+fn)
    else:
        metric = 0.0
        warnings.warn("IoU: Division by zero. Metric is undefined and set to 0")
    return metric
    
# Matthews Correlation Coefficient
def mcc(cMat):
    tn, fp, fn, tp = cMat.ravel()
    tn, fp, fn, tp = castToFloat32(tn, fp, fn, tp)
    numerator = tp*tn-fp*fn
    denominator = (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    if denominator != 0:
        metric = numerator/math.sqrt(denominator)
    else:
        metric = 0.0
        warnings.warn("MCC: Division by zero. Metric is undefined and set to 0")
    return metric

# cMats     = ndarray of shape (n_classes, 2, 2)
# n_metrics = Number of metrics (scalar)
# Returns computed metrics per class, ndarray of shape (n_classes, n_metrics)
def computeMetrics(cMats, n_metrics=6):
    cMats = cMats[1:len(cMats),:,:] #REMOVE LABEL 0 = UNCLASSIFIED
    metrics=torch.zeros(len(cMats),n_metrics,dtype=torch.float32)
    for i, cMat in enumerate (cMats):
        metrics[i][0] = acc(cMat)
        metrics[i][1] = prc(cMat)
        metrics[i][2] = rcl(cMat)
        metrics[i][3] = f1s(cMat)
        metrics[i][4] = iou(cMat)
        metrics[i][5] = mcc(cMat)     
    return metrics

# Weighted Macro-Average model metrics
# metrics                 = metrics per class, ndarray of shape (n_classes, n_metrics)
# classCounts             = Class count wrt dataset, tensor(n_classes).
#                           Call classCount() in dataset/utils.py and
#                           save the class count
# Returns weighted macro-averaged model metrics
def wma(metrics,classCounts):    
    classCounts = classCounts[1:len(classCounts)] #REMOVE LABEL 0 = UNCLASSIFIED
    
    metricsWMA = torch.zeros(len(metrics[1]),dtype=torch.float32)
    
    weights = torch.zeros(len(classCounts),dtype=torch.float32)
    weights = classCounts/classCounts.sum()
    
    # Permute metrics (n_classes, n_metrics) --> (n_metrics, n_classes)
    metrics = torch.permute(metrics,(1,0))
    
    # Multiply each class metric with class weight and sum to get
    # weighted macro-average model metric
    for i, metric in enumerate(metrics):
        metricsWMA[i] = np.multiply(weights,metric).sum()
    
    return metricsWMA # (Accuracy,Precision,Recall,F1-Score,IoU,MCC)

########################### USE FOR VALIDATION ##############################
# Computes Weighted Macro-Average Model IoU
# cMats             = ndarray of shape (n_classes, 2, 2) 
#                     with confusion matrices per class
# classCounts       = Class count wrt dataset, tensor(n_classes).
#                     Call classCount() in dataset/utils.py and
#                     save the class count
def valMetric(cMats,classCounts):
    cMats = cMats[1:len(cMats),:,:] #REMOVE LABEL 0 = UNCLASSIFIED
    classCounts = classCounts[1:len(classCounts)] #REMOVE LABEL 0 = UNCLASSIFIED
    
    weights = torch.zeros(len(classCounts),dtype=torch.float32)
    weights = classCounts/classCounts.sum()
    
    iou_class = torch.zeros(len(classCounts),dtype=torch.float32)
    
    for i, cMat in enumerate (cMats):
        iou_class[i] = iou(cMat)
    
    iou_metric = 0.0
    iou_metric = np.multiply(weights,iou_class).sum()
    
    return iou_metric
############################################################################

############################ Metric Visualization ##########################

# Prints table with metrics per class
# metrics           = ndarray of shape (n_classes, n_metrics) with class metrics
# classCounts       = Class count wrt dataset, tensor(n_classes).
#                     Call classCount() in dataset/utils.py and
#                     save the class count
def printClassMetrics(metrics,classCounts):
        
    classCounts = classCounts[1:len(classCounts)] #REMOVE LABEL 0 = UNCLASSIFIED
    classCounts = classCounts/classCounts.sum() * 100    
    strlist2 = strlist[1:len(strlist)]   #REMOVE LABEL 0 = UNCLASSIFIED
    
    d = {}
    for i in range(len(metrics)):
        d[i] = [classCounts[i], metrics[i][0].item(), metrics[i][1].item(), metrics[i][2].item(), 
                metrics[i][3].item(), metrics[i][4].item(), metrics[i][5].item(), strlist2[i], 
                strlist_ua[i]]

    print("@@ Class Metrics @@\n")

    print("{:<8} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
        'Label', 'UA2018 ID', '% Data','Accuracy','Precision','Recall',
        'F1-Score','IoU','MCC','Description'))
    
    print("------------------------------------------------------------------------------"+
          "------------------------------------------------------------------------------"+
          "----------------------------")

    for l, v in d.items():
        p, acc, prc, rcl, f1s, iou, mcc, dsc, ua = v
        print ("{:<8} {:<15} {:<10.2f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10}".format(
            l+1, ua, p, acc, prc, rcl, f1s, iou, mcc, dsc))

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
    cMats = cMats[1:len(cMats),:,:]      #REMOVE LABEL 0 = UNCLASSIFIED
    strlist2 = strlist[1:len(strlist)]   #REMOVE LABEL 0 = UNCLASSIFIED
     
    labels = ["1","2","3","4","5","6","7","8","9","10","11",
              "12","13","14","15","16","17","18","19","20",
              "21","22","23","24","25","26"]

    fig, axes = plt.subplots(nrows=13, ncols=2, figsize=(20,45))

    group_names = ["True Negative","False Positive","False Negative","True Positive"]

    counter = 0

    for row in axes:
        for col in row:
            group_counts = ["{:,}".format(value) for value in cMats[counter].flatten()]
            group_percentages = ["{0:.2%}".format(value) for value in cMats[counter].flatten()/cMats[counter].sum()]
            labels2 = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
            labels2 = np.asarray(labels2).reshape(2,2)

            sns.set(font_scale=1.1)
            sns.heatmap(cMats[counter], annot=labels2, fmt="", cmap='Blues', ax=col)
            col.set_title("Label " + labels[counter] + " (UA18: " + strlist_ua[counter] + "): " + strlist2[counter], fontsize=11)
            col.set_xlabel("Predicted Label", fontsize=12)
            col.set_ylabel("True Label", fontsize=12)

            counter+=1

    plt.tight_layout()  
    plt.show()

# Prints a n_class x n_class confusion matrix
# yTrue       = tensor([labels])
# yPred       = tensor([predictions])
# CMAP        = The gradient of the values displayed from matplotlib.pyplot.cm
# NORMALIZE   = If False, plot the raw numbers
#               If True, plot the proportions
def printConfusionMatrix(yTrue,yPred,CMAP='Blues',NORMALIZE=True):
    # Filter out label = 0 (Unclassified)
    yTrue_fltr = yTrue[yTrue != 0]
    yPred_fltr = yPred[yTrue != 0]

    LABELS = np.unique(yTrue_fltr)

    cMat = confusion_matrix(y_true=yTrue_fltr,y_pred=yPred_fltr,labels=LABELS)

    plot_confusion_matrix(cm=cMat,target_names=LABELS,cmap=CMAP,normalize=NORMALIZE)

# Source: https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    Given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions (%)
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    h = 16
    w = 15
    fig = plt.figure(figsize=(h, w))
    ax = fig.add_subplot(1,1,1)
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    ax.tick_params(axis="y", left=True, labelleft=True)
    r = np.random.random((h, w))
    imRatio = r.shape[0]/r.shape[1]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.046*imRatio, pad=0.04)
    plt.grid(None)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #cm = cm.astype('float') / cm.sum()

    #thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            if cm[i,j] == np.diag(cm)[i]:
                plt.text(j, i, "{:0.2%}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="lime",
                     fontsize=8, fontweight="roman")

            plt.text(j, i, "{:0.2%}".format(cm[i, j]),
                     horizontalalignment="center",
                     #color="red" if cm[i, j] > thresh else "black",
                     #color="grey",
                     color="red" if (cm[i,j] == cm[i,:].max()) & (cm[i,j] != np.diag(cm)[i]) else "dimgrey",
                     fontsize=8, fontweight="roman")
        else:
            if cm[i,j] == np.diag(cm)[i]:
                plt.text(j, i, "{:0.2%}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="lime",
                    fontsize=8, fontweight="roman")
                    
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     #color="red" if cm[i, j] > thresh else "black",
                     #color="grey",
                     color="red" if (cm[i,j] == cm[i,:].max()) & (cm[i,j] != np.diag(cm)[i]) else "dimgrey",
                     fontsize=8, fontweight="roman")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()