


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


########### Transformations ############# 

# Global percentile Min-Max normalization, better known as RobustScaler
# Less sensitive to outliers than traditional Min-Max normalization
# https://medium.com/@kesarimohan87/data-preprocessing-6c87d27156
# https://www.geeksforgeeks.org/feature-scaling-part-3
# minPer = Min percentile (scalar or tensor)
# maxPer = Max percentile (scalar or tensor)
# Sample = Image patch
# not working
class pNormalize_new(object):
    
    def __init__(self,minPer,maxPer):
        self.minPer = minPer
        self.maxPer = maxPer
    
    def __call__(self,sample):
            
        # According to https://github.com/charlotte-pel/temporalCNN
        sample = sample.transpose(1,3)
        norm = (sample-self.minPer)/(self.maxPer-self.minPer)
        norm = norm.transpose(3,1)
        return norm

class pNormalize(object):
    
    def __init__(self,minPer,maxPer):
        self.minPer = minPer
        self.maxPer = maxPer
        
    def __call__(self,sample):
        
        # According to https://github.com/charlotte-pel/temporalCNN
        norm = (sample-self.minPer[:,None,None])/(self.maxPer[:,None,None]-self.minPer[:,None,None])
        return norm