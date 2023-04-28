from torchvision.models import resnet34, ResNet34_Weights
import torch
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

class DataCollector:
    def __init__(self, height, width, classes,skin_tones=None,mask_opts = None,):
        self.height = height
        self.width = width
        self.classes = classes
        self.num_images = 0
        self.skin_tones = skin_tones
        if mask_opts!= None:
            self.mask_bool = True
            self.mask_opts = mask_opts
            self.mask_size = mask_opts["mask_size"]
            self.mask_pix_vals = mask_opts["skintone_label"]
        else:
            self.mask_bool = False
        for _, class_images in self.classes.values():
            self.num_images += class_images
    
    ### Scrape files to collect the data
    def collect(self, filepath, val_size=0.2, test_size=0.2, random_state=42):
        X = []
        y = []
        self.datasets = {}
        
        # Default weight data
        mean = ResNet34_Weights.DEFAULT.transforms().mean
        std = ResNet34_Weights.DEFAULT.transforms().std
        
        for image_class, (class_id, class_images) in self.classes.items():
            for i in range(1, 1 + class_images):
                image = np.array(Image.open(f'{filepath}/{image_class}/{image_class}{i}.jpg'))
                for i in range(len(image)):
                    image[i]= image[i] - mean
                    image[i]= image[i] / std
                X.append(image)
                y.append(class_id)
        
        # Store data as arrays
        X = np.array(X)
        self.X = X.reshape((self.num_images, 3, self.height, self.width))
        self.y = np.array(y)
        
        print("Store in X and y")
        
        # Split data into train, validation, and test
        train_X, valtest_X, train_y, valtest_y = train_test_split(self.X, self.y, test_size=(val_size+test_size), random_state=random_state)
        
        if self.mask_bool:
            train_X = train_X.reshape((len(train_X), self.height, self.width,3))   
            for i in range(len(train_X)):
                j = np.random.randint(self.height-self.mask_size)
                k = np.random.randint(self.width-self.mask_size)
                train_X[j:j+self.mask_size,k:k+self.mask_size] = self.mask_pix_vals
            train_X = train_X.reshape((len(train_X),3, self.height, self.width))   
        
        
        
        val_X, test_X, val_y, test_y = train_test_split(valtest_X, valtest_y, test_size=(test_size/(test_size+val_size)), random_state=random_state)
        
        self.datasets['train'] = torch.utils.data.TensorDataset(torch.FloatTensor(train_X), torch.LongTensor(train_y))
        self.datasets['val'] = torch.utils.data.TensorDataset(torch.FloatTensor(val_X), torch.LongTensor(val_y))
        self.datasets['test'] = torch.utils.data.TensorDataset(torch.FloatTensor(test_X), torch.LongTensor(test_y))
        
        return self.datasets