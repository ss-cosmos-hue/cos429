from torchvision.models import resnet34, ResNet34_Weights
import torch.nn as nn
import torch
from torchsummary import summary
import torch.optim as optim
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from configurations import configurations

def collect_data():
    H = configurations.height
    W = configurations.width
    num_images = np.sum(list(configurations.class_sizes.values()))
    X = np.zeros((num_images, H, W, 3))
    y = np.zeros((num_images))
    
    class_label = 0
    idx = 0
    for categ in configurations.classes:
        i = 0
        while True:
            i += 1
            try:
                image = np.array(Image.open(f'data/trashnet/{categ}/{categ}{i}.jpg'))
                X[idx]=image
                y[idx]=class_label
                idx += 1
            except:
                class_label += 1
                break
    
    X = X.reshape((num_images, 3, H, W))
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