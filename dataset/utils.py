import torch

# data_loader     = DataLoader(dataset)
# n_classes       = 28 (number of classes including unclassified class)

# Returns class count wrt dataset: n_classes_dataset, tensor(n_classes)
#                     wrt sample:  n_classes_sample,  tensor(n_samples, n_classes)
def classCount(data_loader,n_classes=28):

    dataiter = iter(data_loader)
    
    n_classes_dataset = torch.zeros(n_classes,dtype=torch.int32)
    
    n_classes_sample = torch.zeros(len(dataiter),n_classes,dtype=torch.int32)
    
    # i = samples/patches in dataset, j = class in sample/patch
    # j = 0 => label 0, j = 1 => label 1.... j = 27 => label 27 
    for i, (_, labels) in enumerate(dataiter):
        classes,count = labels.unique(return_counts=True)
        for j in classes:
            idx = (classes==j).nonzero().item()
            classCount = count[idx].item()
            n_classes_dataset[j] += classCount
            n_classes_sample[i][j] = classCount
            
    return n_classes_dataset, n_classes_sample

# data_loader     = DataLoader(dataset)
# n_classes       = 28 (number of classes including unclassified class)

# Returns cross entropy loss weights wrt dataset: n_classes_dataset, tensor(n_classes)
#                                    wrt sample:  n_classes_sample,  tensor(n_samples, n_classes)

def crossEntropyLossWeights(data_loader,n_classes=28):

    dataiter = iter(data_loader)
    
    n_classes_dataset = torch.zeros(n_classes,dtype=torch.int32)
    n_classes_sample = torch.zeros(len(dataiter),n_classes,dtype=torch.int32)
    n_classes_dataset, n_classes_sample = classCount(data_loader,n_classes)
    
    class_weights_dataset = torch.zeros(n_classes,dtype=torch.float32)
    class_weights_sample = torch.zeros(len(dataiter),n_classes,dtype=torch.float32)
    
    ############### Weights wrt sample
    
    # Set class weight = 0 for label = 0
    # Inf control. Init torch.zero not actually "zero"
    # i = samples/patches in dataset, j = class in sample/patch
    # j = 0 => label 0, j = 1 => label 1.... j = 27 => label 27 
    for i in range(len(n_classes_sample)):
        for j in range(len(n_classes_sample[i])):
            if j == 0:
                class_weights_sample[i][j] = 0
            elif n_classes_sample[i][j].item() != 0:
                class_weights_sample[i][j] = torch.max(n_classes_sample[i])/n_classes_sample[i][j]

    ################ Weights wrt dataset
                
    # Set class weight = 0 for label = 0
    # Inf control. Init torch.zero not actually "zero"
    # i = class in sample/patch
    # i = 0 => label 0, i = 1 => label 1.... i = 27 => label 27 
    for i, count in enumerate(n_classes_dataset):
        if i == 0:
            class_weights_dataset[i] = 0
        elif count.item() != 0:
            class_weights_dataset[i] = torch.max(n_classes_dataset)/n_classes_dataset[i]
    
    return class_weights_dataset, class_weights_sample
    
########### Transformations ############# 

    # Global percentile Min-Max normalization, better known as RobustScaler
    # Less sensitive to outliers than traditional Min-Max normalization
    # https://medium.com/@kesarimohan87/data-preprocessing-6c87d27156
    # https://www.geeksforgeeks.org/feature-scaling-part-3
    # minPer = Min percentile (scalar or tensor)
    # maxPer = Max percentile (scalar or tensor)
    # Sample = Image patch tensor ([channels, H,W])
    
class pNormalize(object):
    
    def __init__(self,minPer,maxPer):
        self.minPer = minPer
        self.maxPer = maxPer
    
    def __call__(self,sample):
            
        # According to https://github.com/charlotte-pel/temporalCNN
        sample = sample.permute(2,1,0)
        norm = (sample-self.minPer)/(self.maxPer-self.minPer)
        norm = norm.permute(2,1,0)
        
        # Above is easier to read
        # norm = (sample-self.minPer[:,None,None])/(self.maxPer[:,None,None]-self.minPer[:,None,None])
        
        return norm
