from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import math
import warnings
import itertools
import io
import os


# Class descriptions. See classDict.py!
n_classes_dsc = 28
 
strlist = [{}]*n_classes_dsc 
strlist[0] = "Unclassified"
strlist[1] = "Continuous Urban Fabric"
strlist[2] = "Discontinuous Dense Urban Fabric"
strlist[3] = "Discontinuous Medium Density Urban Fabric"
strlist[4] = "Discontinuous Low Density Urban Fabric"
strlist[5] = "Discontinuous Very Low Density Urban Fabric"
strlist[6] = "Isolated Structures"
strlist[7] = "Industrial, commercial, public, military and private units"
strlist[8] = "Fast transit roads and associated land"
strlist[9] = "Other roads and associated land"
strlist[10] = "Railways and associated land"
strlist[11] = "Port areas"
strlist[12] = "Airports"
strlist[13] = "Mineral extraction and dump sites"
strlist[14] = "Construction sites"
strlist[15] = "Land without current use"
strlist[16] = "Green urban areas"
strlist[17] = "Sports and leisure facilities"
strlist[18] = "Arable land (annual crops)"
strlist[19] = "Permanent crops (vineyards, fruit trees, olive groves)"
strlist[20] = "Pastures"
strlist[21] = "Complex and mixed cultivation patterns"
strlist[22] = "Orchards at the fringe of urban classes"
strlist[23] = "Forests"
strlist[24] = "Herbaceous vegetation associations (natural grassland, moors...)"
strlist[25] = "Open spaces with little or no vegetations (beaches, dunes, bare rocks, glaciers)"
strlist[26] = "Wetland"
strlist[27] = "Water bodies"

# Urban Atlas 2018 IDs
n_classes_ua = 27

strlist_ua = [{}]*n_classes_ua
strlist_ua[0] = "11100"
strlist_ua[1] = "11210"
strlist_ua[2] = "11220"
strlist_ua[3] = "11230"
strlist_ua[4] = "11240"
strlist_ua[5] = "11300"
strlist_ua[6] = "12100"
strlist_ua[7] = "12210"
strlist_ua[8] = "12220"
strlist_ua[9] = "12230"
strlist_ua[10] = "12300"
strlist_ua[11] = "12400"
strlist_ua[12] = "13100"
strlist_ua[13] = "13300"
strlist_ua[14] = "13400"
strlist_ua[15] = "14100"
strlist_ua[16] = "14200"
strlist_ua[17] = "21000"
strlist_ua[18] = "22000"
strlist_ua[19] = "23000"
strlist_ua[20] = "24000"
strlist_ua[21] = "25000"
strlist_ua[22] = "31000"
strlist_ua[23] = "32000"
strlist_ua[24] = "33000"
strlist_ua[25] = "40000"
strlist_ua[26] = "50000"

############################# HOW TO COMPUTE MODEL METRICS ########################
#    1. computeConfMats()
#    2. computeMetrics() (use returned class metrics in step 4)
#    3. classCount()     (in dataset/utils.py, use returned class counts in step 4)
#    4. wma()
###################################################################################

# yTrue       = tensor(B,H,W), B = Batch Size, H = Height, W = Width
# yPred       = tensor(B,H,W), B = Batch Size, H = Height, W = Width
# n_classes   = number of classes (scalar)
# Returns ndarray of shape (n_classes-1, 2, 2) with confusion matrices per class.
# Removes class 0 = UNCLASSIFIED
def computeConfMats(yTrue,yPred):
    LABELS = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,
                       23,24,25,26,27])
    
    # Flatten dimensions BxHxW --> B*H*W
    yTrue = yTrue.reshape(-1)
    yPred = yPred.reshape(-1)
    
    # Remove LABEL 0 = UNCLASSIFIED
    yTrue_fltr = yTrue[yTrue != 0]
    yPred_fltr = yPred[yTrue != 0]
    
    cMats = multilabel_confusion_matrix(y_true=yTrue_fltr,y_pred=yPred_fltr,labels=LABELS)
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
        #warnings.warn("Precision: Division by zero. Metric is undefined and set to 0")
    return metric
    
# Recall
def rcl(cMat):
    tn, fp, fn, tp = cMat.ravel()
    tn, fp, fn, tp = castToFloat32(tn, fp, fn, tp)
    if (tp+fn) != 0:
        metric = tp/(tp+fn)
    else:
        metric = 0.0
        #warnings.warn("Recall: Division by zero. Metric is undefined and set to 0")
    return metric

# F1-Score
def f1s(cMat):
    tn, fp, fn, tp = cMat.ravel()
    tn, fp, fn, tp = castToFloat32(tn, fp, fn, tp)
    if (fp+tp+tp+fn) != 0:
        metric = (2*tp)/(fp+tp+tp+fn)
    else:
        metric = 0.0
        #warnings.warn("F1-Score: Division by zero. Metric is undefined and set to 0")
    return metric

# Intersection over Union
def iou(cMat):
    tn, fp, fn, tp = cMat.ravel()
    tn, fp, fn, tp = castToFloat32(tn, fp, fn, tp)
    if (fp+tp+fn) != 0:
        metric = tp/(fp+tp+fn)
    else:
        metric = 0.0
        #warnings.warn("IoU: Division by zero. Metric is undefined and set to 0")
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
        #warnings.warn("MCC: Division by zero. Metric is undefined and set to 0")
    return metric

# cMats     = ndarray of shape (n_classes, 2, 2)
# n_metrics = Number of metrics (scalar)
# Returns computed metrics per class, ndarray of shape (n_classes, n_metrics)
def computeClassMetrics(cMats, n_metrics=6):
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

# Converts BytesIO object with plot to tensor and writes to TensorBoard
# buf   = BytesIO object
# title = Image name (String)
# path  = Path-string
def toTensorboard(buf,title,path):
    tb_writer = SummaryWriter(os.path.join(path,title))
    buf.seek(0)
    im = Image.open(buf)
    im = torchvision.transforms.ToTensor()(im)
    tb_writer.add_image(title, im)
    buf.close()
    tb_writer.flush()
    tb_writer.close()

# Prints table with metrics per class
# metrics           = ndarray of shape (n_classes, n_metrics) with class metrics
# classCounts       = Class count wrt dataset, tensor(n_classes).
#                     Call classCount() in dataset/utils.py and
#                     save the class count
# title       = Image name (String), use when printing to Tensorboard
# path        = Path-string, use when printing to Tensorboard
# TB          = if False output print to standard out, if True plots image to Tensorboard

def printClassMetrics(metrics,classCounts,TB=False,title="Class_Metrics",path="runs/"):   
    classCounts = classCounts[1:len(classCounts)] #REMOVE LABEL 0 = UNCLASSIFIED
    classCounts = classCounts/classCounts.sum() * 100
    
    lines = ['\n \n',title, "@@ Class Metrics @@\n","{:<8} {:<15} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            'Label', 'UA2018 ID', '% Data','Accuracy','Precision','Recall',
            'F1-Score','IoU','MCC','Description'),"------------------------------------------------------------------------------"+
            "------------------------------------------------------------------------------"+
            "----------------------------"]

    if TB == False:    
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
            print ("{:<8} {:<15} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10}".format(
                l+1, ua, p, acc, prc, rcl, f1s, iou, mcc, dsc))
            lines.append("{:<8} {:<15} {:<10.2f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10}".format(
                l+1, ua, p, acc, prc, rcl, f1s, iou, mcc, dsc))

        with open(os.path.join(path,'metrics.txt'), 'a') as f:
            f.write('\n'.join(lines))

        
    else:
        strlist3 = strlist
        strlist3[7] = "Industrial,commercial,public,military & private units"
        strlist3[19] = "Permanent crops"
        strlist3[24] = "Herbaceous vegetation associations"
        strlist3[25] = "Open spaces with little or no vegetations"
        strlist3 = strlist3[1:len(strlist3)]   #REMOVE LABEL 0 = UNCLASSIFIED
        
        d = {}
        for i in range(len(metrics)):
            d[i] = [classCounts[i], metrics[i][0].item(), metrics[i][1].item(), metrics[i][2].item(), 
                    metrics[i][3].item(), metrics[i][4].item(), metrics[i][5].item(), strlist3[i], 
                    strlist_ua[i]]
            
        df = pd.DataFrame(columns=['Label','UA2018 ID', '% Data', 'Accuracy', 'Precision', 
                           'Recall', 'F1-Score', 'IoU', 'MCC'])
        
        for l, v in d.items():
            p, acc, prc, rcl, f1s, iou, mcc, dsc, ua = v

            dataRow = pd.DataFrame({'Label':[l+1], 'UA2018 ID':ua, '% Data':[p.item()], 'Accuracy':[acc], 'Precision':[prc], 
                        'Recall':[rcl], 'F1-Score':[f1s], 'IoU':[iou], 'MCC':[mcc],'Description':dsc})

            dataRow = dataRow.round({'% Data':2, 'Accuracy':4, 'Precision':4, 'Recall':4, 'F1-Score':4, 'IoU':4, 'MCC':4})

            df = pd.concat([df,dataRow], ignore_index=True)           
            
        N = 20
        fig = plt.figure(figsize=(20,2+N/3))
        ax = plt.subplot(111, frame_on=False) # No visible frame
        ax.xaxis.set_visible(False)  # Hide the x axis
        ax.yaxis.set_visible(False)  # Hide the y axis
        
        table = ax.table(cellText=df.values, colLabels=df.keys(),
                 loc='center',cellLoc="left",colLoc="left",edges="closed")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        toTensorboard(buf,title,path)   

# Prints weighted macro-averaged model metrics 
# metrics = ndarray with model metrics of shape (n_metrics)
# title       = Image name (String), use when printing to Tensorboard
# path        = Path-string, use when printing to Tensorboard
# TB          = if False output print to standard out, if True plots image to Tensorboard
def printModelMetrics(metrics,TB=False,title="Model_Metrics",path="runs/"):
    strlistM = [{}]*len(metrics)
    strlistM[0] = "Accuracy"
    strlistM[1] = "Precision"
    strlistM[2] = "Recall"
    strlistM[3] = "F1-Score"
    strlistM[4] = "Intersection over Union"
    strlistM[5] = "Matthews Correlation Coefficient"


    lines = ['\n \n',title, "@@ Weighted Macro-Average Model Metrics @@\n"]
    for i in range(len(metrics)):
        lines.append(f'{strlistM[i]:<35}: {metrics[i].item():.4f}')
    with open(os.path.join(path,'metrics.txt'), 'a') as f:
        f.write('\n'.join(lines))

    if TB == False:    
        print("@@ Weighted Macro-Average Model Metrics @@\n")
        
        for i in range(len(metrics)):
            print(f'{strlistM[i]:<35}: {metrics[i].item():.4f}')
            
    else:
        d = {strlistM[0]: metrics[0].item(), strlistM[1]: metrics[1].item(), strlistM[2]: metrics[2].item(), 
             strlistM[3]: metrics[3].item(), strlistM[4]: metrics[4].item(), strlistM[5]: metrics[5].item()}
        df = pd.DataFrame(data=d, index=[0])
            
        nrows = 0
        ncols = 0

        wcell = 14
        hcell = 15

        wpad = 0.3
        hpad = 1.2

        fig = plt.figure(figsize=(2*wcell+wpad, nrows*hcell+hpad))
        ax = plt.subplot(111, frame_on=False) # No visible frame
        ax.xaxis.set_visible(False)  # Hide the x axis
        ax.yaxis.set_visible(False)  # Hide the y axis
        ax.table(cellText=np.round(df.values,decimals=4), colLabels=df.keys(),
        loc='center',cellLoc="center",colLoc="center",edges="open")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        toTensorboard(buf,title,path) 

# Plots confusion matrices per class        
# cMats = ndarray of shape (n_classes, 2, 2) with confusion matrices per class
# title       = Image name (String), use when printing to Tensorboard
# path        = Path-string, use when printing to Tensorboard
# TB          = if False output print to standard out, if True plots image to Tensorboard
def plotConfusionMatrices(cMats, TB=False,title="Confusion_Matrices",path="runs/"):
    strlist2 = strlist
    strlist2[7] = "Industrial,commercial,public,military & private units"
    strlist2[19] = "Permanent crops"
    strlist2[24] = "Herbaceous vegetation associations"
    strlist2[25] = "Open spaces with little or no vegetations"
    strlist2 = strlist2[1:len(strlist2)]   #REMOVE LABEL 0 = UNCLASSIFIED
     
    labels = ["1","2","3","4","5","6","7","8","9","10","11",
              "12","13","14","15","16","17","18","19","20",
              "21","22","23","24","25","26","27"]

    fig, axes = plt.subplots(nrows=9, ncols=3, figsize=(20,45))

    group_names = ["True Negative","False Positive","False Negative","True Positive"]

    counter = 0

    for row in axes:
        for col in row:
            group_counts = ["{:,}".format(value) for value in cMats[counter].flatten()]
            group_percentages = ["{0:.4%}".format(value) for value in cMats[counter].flatten()/cMats[counter].sum()]
            labels2 = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
            labels2 = np.asarray(labels2).reshape(2,2)

            sns.set(font_scale=1.1)
            sns.heatmap(cMats[counter], annot=labels2, fmt="", cmap='Blues', ax=col)
            col.set_title("Label " + labels[counter] + " (UA18: " + strlist_ua[counter] + "): " + strlist2[counter], fontsize=10)
            col.set_xlabel("Predicted Label", fontsize=12)
            col.set_ylabel("True Label", fontsize=12)

            counter+=1

    plt.tight_layout()
    
    plt.savefig(os.path.join(path,'cMats.png'))
    
    if TB == False:
        plt.show()
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        toTensorboard(buf,title,path)

# Plots a N_CLASS X N_CLASS confusion matrix
# yTrue       = tensor([labels])
# yPred       = tensor([predictions])
# CMAP        = The gradient of the values displayed from matplotlib.pyplot.cm
# NORMALIZE   = If False, plot the raw numbers
#               If True, plot the proportions
# title       = Image name (String), use when printing to Tensorboard
# path        = Path-string, use when printing to Tensorboard
# TB          = if False output print to standard out, if True plots image to Tensorboard
def plotConfusionMatrix(yTrue,yPred,CMAP='Blues',NORMALIZE=True,TB=False,title="Confusion_Matrix",path="runs/"):
    # Filter out label = 0 (Unclassified)
    yTrue_fltr = yTrue[yTrue != 0]
    yPred_fltr = yPred[yTrue != 0]

    LABELS = np.unique(yTrue_fltr)

    cMat = confusion_matrix(y_true=yTrue_fltr,y_pred=yPred_fltr,labels=LABELS)

    plot_confusion_matrix(cm=cMat,target_names=LABELS,cmap=CMAP,normalize=NORMALIZE,tb=TB,title=title,path=path)

# Source: https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
def plot_confusion_matrix(cm,
                          target_names,
                          cmap=None,
                          normalize=True,
                          tb=False,
                          title="confMat",
                          path="runs/confusion_matrix"):
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
    w = 16
    fig = plt.figure(figsize=(h, w))
    ax = fig.add_subplot(1,1,1)
    ax.tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)
    ax.tick_params(axis="y", left=True, labelleft=True)
    ax.spines['bottom'].set_color('black')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    r = np.random.random((h, w))
    imRatio = r.shape[0]/r.shape[1]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title('Confusion matrix')
    plt.title("Predicted label", fontsize=15)
    plt.colorbar(fraction=0.046*imRatio, pad=0.04)
    plt.grid(False)

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)
        
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #cm = cm.astype('float') / cm.sum()

    #thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    diag = np.diag_indices_from(cm)
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            if (i == diag[0][j]) & (j == diag[1][j]):
                plt.text(i, j, "{:0.3%}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="lime",
                     fontsize=8, fontweight="normal")

            plt.text(i, j, "{:0.3%}".format(cm[i, j]),
                     horizontalalignment="center",
                     #color="red" if cm[i, j] > thresh else "black",
                     #color="grey",
                     color="red" if (cm[i,j] == cm[:,j].max()) & (cm[i,j] > cm[diag[0][j],diag[1][j]]) else "dimgrey",
                     fontsize=8, fontweight="normal")
        else:
            if (i == diag[0][j]) & (j == diag[1][j]):
                plt.text(i, j, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="lime",
                    fontsize=8, fontweight="normal")
                    
            plt.text(i, j, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     #color="red" if cm[i, j] > thresh else "black",
                     #color="grey",
                     color="red" if (cm[i,j] == cm[:,j].max()) & (cm[i,j] > cm[diag[0][j],diag[1][j]]) else "dimgrey",
                     fontsize=8, fontweight="normal")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.savefig(os.path.join(path,'cMat.png'))
    
    if tb == False:
        plt.show()
    else:
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        toTensorboard(buf,title,path)


# Plots a Sample RGB,LABEL,PRED,and corecctly predicted
# plot_sample(pred[i,:,:],labels[i,:,:], images[i,0:3,:,:]  ) 
# pred       = tensor([prediction])
# label      = tensor([label])
# rgb        = tensor([RGB channels of X])
# classMax   = highest class value (27)
# classMin   = lowest class value (0)
# fig_scale  = for size scaling of figures
# path        = Path-string, where to save figure, if save_fig = True

def plot_sample(pred,labl,rgb,classMax=27,classMin=0,fig_scale=3,save_fig=True,path=None,fn='sample.png',dpi=150):
    
    corrPred=torch.eq(pred,labl)
    predratio = torch.sum(corrPred)/corrPred.numel()*100
    cmap = plt.get_cmap('Spectral', classMax - classMin + 1)

    nrow = 1
    ncol = 4
    
    gridspec_kw ={ 
                "wspace": 0.0,
                "hspace": 0.0, 
                "top": 1.-0.5/(fig_scale*nrow+1),
                "bottom": 0.5/(fig_scale*nrow+1), 
                "left": 0.5/(fig_scale*ncol+1), 
                "right": 1-0.5/(fig_scale*ncol+1)
                }
    
    fig_kw={
        "figsize": (fig_scale*ncol+1,fig_scale*nrow+1)
        }

    fig, axs =plt.subplots(nrow,ncol,gridspec_kw=gridspec_kw , **fig_kw)

    # Adds a subplot at the 1st column
    axs[0].imshow(rgb.permute(1,2,0).cpu().numpy())
    axs[0].axis('off')
    axs[0].set_title('RGB')
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])

    # Adds a subplot at the 2nd column
    map=axs[1].imshow(labl.cpu().numpy(),cmap=cmap,vmin=(classMin-0.5),vmax=(classMax+0.5))          
    axs[1].axis('off')
    axs[1].set_title("Label (%d classes)"%(len(torch.unique(labl))))
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])

    # Adds a subplot at the 3rd column
    axs[2].imshow(pred.cpu().numpy(),cmap=cmap,vmin=(classMin-0.5),vmax=(classMax+0.5))
    axs[2].axis('off')
    axs[2].set_title("Prediction (%d classes)"%(len(torch.unique(pred))))
    axs[2].set_xticklabels([])
    axs[2].set_yticklabels([])

    # Adds a subplot at the 4th column
    axs[3].imshow(corrPred.cpu().numpy(),cmap="binary",vmin=0,vmax=1)
    axs[3].axis('off')
    axs[3].set_title("%4.2f %% Predicted Correct"%(predratio)) 
    axs[3].set_xticklabels([])
    axs[3].set_yticklabels([])

    fig.colorbar(map, ax=axs, cmap=cmap, ticks=np.arange(classMin,classMax + 1),shrink=0.6,location='bottom')

    if save_fig:
        if path:
            
            if not os.path.exists(path):
                os.makedirs(path)

            path= os.path.join(path,fn)
        else:
            path= fn

        plt.savefig(path,dpi=dpi)
    else:
        plt.show()



# Plots a batch of RGB,LABEL,PRED,and corecctly predicted
# plot_batch(pred,labels, images) 
# pred       = tensor([prediction])
# label      = tensor([label])
# rgb        = tensor([RGB channels of X])
# classMax   = highest class value (27)
# classMin   = lowest class value (0)
# fig_scale  = for size scaling of figures
# path        = Path-string, where to save figure, if save_fig = True
def plot_batch(pred,labl,rgb,classMax=27,classMin=0,fig_scale=1,save_fig=True,path=None,fn='batch_sample.png',dpi=150):
    
    # correct predicted % per patch
    predRatio = torch.sum(torch.eq(pred,labl),dim=[1,2])/pred[0,:,:].numel()*100
    corrPred=torch.eq(pred,labl)
    
    cmap = plt.get_cmap('Spectral', classMax - classMin + 1)

    nrow = pred.shape[0]
    ncol = 4
    
    ymax=pred.shape[-2]

    gridspec_kw ={ 
                "wspace": 0.0,
                "hspace": 0.0, 
                "top": 1.-0.5/(fig_scale*nrow+1),
                "bottom": 0.5/(fig_scale*nrow+1), 
                "left": 0.5/(fig_scale*ncol+1), 
                "right": 1-0.5/(fig_scale*ncol+1)
                }
    
    fig_kw={
        "figsize": (fig_scale*ncol+1,fig_scale*nrow+1)
        }

    fig, axs =plt.subplots(nrow+1,ncol, gridspec_kw=gridspec_kw , **fig_kw)

    for i in range(nrow):
        # Adds a subplot at the 1st column
        axs[i,0].imshow(rgb[i,0:3,:,:].permute(1,2,0).cpu().numpy())
        axs[i,0].axis('off')
        axs[i,0].set_xticklabels([])
        axs[i,0].set_yticklabels([])

        # Adds a subplot at the 2nd column
        map=axs[i,1].imshow(labl[i,:,:].cpu().numpy(),cmap=cmap,vmin=(classMin-0.5),vmax=(classMax+0.5))          
        axs[i,1].axis('off')
        axs[i,1].set_xticklabels([])
        axs[i,1].set_yticklabels([])

        # Adds a subplot at the 3rd column
        axs[i,2].imshow(pred[i,:,:].cpu().numpy(),cmap=cmap,vmin=(classMin-0.5),vmax=(classMax+0.5))
        axs[i,2].axis('off')
        axs[i,2].set_xticklabels([])
        axs[i,2].set_yticklabels([])

        # Adds a subplot at the 4th column
        perc = str(round(predRatio[i].item(),2)) + '%'
        axs[i,3].imshow(corrPred[i,:,:].cpu().numpy(),cmap="binary",vmin=0,vmax=1)
        axs[i,3].axis('off')
        axs[i,3].text(0.01,ymax*0.99,perc,color='lime')
        axs[i,3].set_xticklabels([])
        axs[i,3].set_yticklabels([])

    # for colorbar
    for j in range(ncol):
        axs[i+1,j].axis('off')
    fig.colorbar(map, ax=axs[i+1,:4], cmap=cmap, ticks=np.arange(classMin,classMax + 1),shrink=0.6,location='bottom')

    if save_fig:
        if path:
            
            if not os.path.exists(path):
                os.makedirs(path)
            path= os.path.join(path,fn)
        else:
            path= fn

        plt.savefig(path,dpi=dpi)
    else:   
        plt.show()
