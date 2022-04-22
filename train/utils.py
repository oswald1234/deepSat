from tqdm.notebook import tqdm

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
