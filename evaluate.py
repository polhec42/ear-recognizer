import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from prepare_dataset import get_train_data, get_test_data
import cv2


def calculate_auc(y_values):
    auc = 0

    for v in y_values:
        auc += v*0.01

    return auc

def index_of_first_n(sez, n):
    
    result = [0]*n
    idx = [0]*n

    for i in range(0,len(sez)):
        el = sez[i]
        if el > min(result):
            index = result.index(min(result))
            result[index] = el
            idx[index] = i

    return idx

def evaluate_n(n, test_data, model):
    correct_count, all_count = 0, 0

    for batch_objects in test_data:
        images = batch_objects['image']
        labels = batch_objects['class']
        for i in range(len(labels)):
            
            img = images[i]
            # Turn off gradients to speed up this part
            with torch.no_grad():
                logps = model(images.cuda())

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.cpu().numpy()[i])

            true_label = labels.numpy()[i]

            sez = index_of_first_n(probab, n)
 
            for index in sez:
                if(true_label == index):
                    correct_count += 1


            all_count += 1

    return correct_count/all_count

if __name__ == "__main__":

    test_data = get_test_data()
    

    model = torch.load('./best.pt')
    model.eval()

    batch_objects = next(iter(test_data))
    images = batch_objects['image']
    labels = batch_objects['class']
    logps = model(images.cuda())
    """
    def view_classify(img, ps):
        ''' Function for viewing an image and it's predicted classes.
        '''
        ps = ps.cpu().data.numpy().squeeze()[0]

        fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
        
        ax1.imshow(img.permute(1,2,0).numpy().squeeze().astype('uint8'))
        ax1.axis('off')
        ax2.barh(np.arange(100), ps)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(100))
        ax2.set_yticklabels(np.arange(100))
        ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()
        plt.show()
    # Output of the network are log-probabilities, need to take exponential for probabilities

    ps = torch.exp(logps)
    probab = list(ps.cpu().detach().numpy()[0])
    print("Predicted ear label = ", probab.index(max(probab))+1, labels.numpy()[0]+1)
    view_classify(images[0], ps)
    """

    correctly_classified = []

    rank = 5

    for n in range(1,rank+1):

        correctly_classified.append(evaluate_n(n, test_data, model))

    print(correctly_classified)

    if rank == 100:
        print(calculate_auc(correctly_classified))

    # Plotting CMC curves
    #plt.plot(range(1,101), correctly_classified)
    #plt.savefig("CMC-original.pdf")