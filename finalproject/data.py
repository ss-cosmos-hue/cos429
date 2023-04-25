from torchvision.models import resnet34, ResNet34_Weights
import torch.nn as nn
import torch
from torchsummary import summary
import torch.optim as optim
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from configurations import configurations

def collect_data():
    H = configurations.height
    W = configurations.width
    classes = configurations.classes
    class_sizes = configurations.class_sizes
    class_labels = configurations.class_labels
    num_images = np.sum(list(configurations.class_sizes.values()))
    
#     X = np.zeros((num_images, H, W, 3))
#     y = np.zeros((num_images))
    
#     class_label = 0
#     idx = 0
#     for categ in configurations.classes:
#         i = 0
#         while True:
#             i += 1
#             try:
#                 image = np.array(Image.open(f'data/trashnet/{categ}/{categ}{i}.jpg'))
#                 X[idx]=image
#                 y[idx]=class_label
#                 idx += 1
#             except:
#                 class_label += 1
#                 break
    
#     X = X.reshape((num_images, 3, H, W))
    X = []
    y = []
    mean = ResNet34_Weights.DEFAULT.transforms().mean
    std = ResNet34_Weights.DEFAULT.transforms().std
    #update X and y
    total_count  = 0
    for categ in classes:
        n_elem = class_sizes[categ]
        for i in range(1,1+n_elem):
            if total_count%100 == 0:
                print(f'total_count = {total_count}')
            # X[total_count] = np.array(Image.open(f'trashnet/dataset-resized/{categ}/{categ}{i}.jpg'))
            image = np.array(Image.open(f'trashnet/dataset-resized/{categ}/{categ}{i}.jpg'))
            # image = np.copy(plt.imread(f'trashnet/dataset-resized/{categ}/{categ}{i}.jpg'))
            for i in range(len(image)):
                image[i]=image[i]-mean
                image[i]=image[i]/std
            X.append(image)
            y.append(class_labels[categ])
            total_count += 1
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((num_images,3,H,W))
    print("store in X and y")
    return X,y

def gen_dataset(X,y,val_size:float,test_size:float,random_state = 42):
    train_X,test_X,train_y,test_y = train_test_split(X, y, test_size=(val_size+test_size), random_state=random_state)
    val_X,test_X ,val_y,test_y = train_test_split(test_X, test_y, test_size=(test_size/(test_size+val_size)), random_state=random_state)
    def gen_torch_dataset(X,y):
        return torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    datasets = {'train':gen_torch_dataset(train_X,train_y),
                'val':gen_torch_dataset(val_X,val_y),
                'test':gen_torch_dataset(test_X,test_y)}
    return datasets