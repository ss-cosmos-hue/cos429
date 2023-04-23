from torchvision.models import resnet34, ResNet34_Weights
import torch.nn as nn
import torch
from torchsummary import summary
import torch.optim as optim
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from configurations import configurations

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Model:
    def __init__(self, **kwargs):
        self.height = configurations.height
        self.width = configurations.width
        self.num_classes = configurations.class_nums
        self.dataloaders = {}
        self.learning_rate = kwargs.get('lr')
        self.batch_size = kwargs.get('bs')
        self.momentum = kwargs.get('rho')
        self.decay_step = kwargs.get('gamma_step')
        self.decay_proportion = kwargs.get('gamma')
        self.n_activelayer = kwargs.get('n_active_layers')
        self.replace = kwargs.get('if_replace')
        self.num_workers = kwargs.get('workers')
        self.num_epochs = kwargs.get('epochs')
        self.mask = kwargs.get('mask')
        self.unique_filename = kwargs.get('unique_filename')
        self.configure_model()

    def configure_model(self):
        # Use library model
        self.model = resnet34(weights = ResNet34_Weights.DEFAULT)

        if self.replace == True:
            # Fully-connected layer with correct number of classes
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, self.num_classes)
        else:
            self.model.add_module('fc2',nn.Linear(1000, self.num_classes))#not sure if this will work
            #or maybe self.model = nn.Sequential

        # Specify device preference
        self.model = self.model.to(device)

        # Freeze parameters except for last n_activelayer layer
        for param in self.model.parameters():
            param.requires_grad = False
        for i in range(self.n_activelayer):    
            active_layer = list(self.model.children())[-(i+1)]
            for param in active_layer.parameters():
                param.requires_grad = True
        
        # Print summary of the model
        # summary(self.model, (3, self.height, self.width))
    
    def construct_data(self, datasets):
        # Generate data loaders
        self.dataloaders['train'] = torch.utils.data.DataLoader(dataset=datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.dataloaders['val'] = torch.utils.data.DataLoader(dataset=datasets['val'], batch_size = len(datasets['val']), shuffle = False)
        self.dataloaders['test'] = torch.utils.data.DataLoader(dataset=datasets['test'], batch_size = len(datasets['test']), shuffle = False)

        # Define loss optimization
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_ft = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer_ft, step_size=self.decay_step, gamma=self.decay_proportion)   
    
    def saveinfo(self, model, training_loss, training_accuracy, validation_loss, validation_accuracy):
        #save kwargs, model,statistics
        unique_filename = self.unique_filename
        with open(f'modelinfo/{unique_filename}',"w") as file:
            for key, value in self.kwargs.items():
                file.write("{}: {}\n".format(key, value))
        torch.save(model,f'model/{unique_filename}')

        np.save(f'statistics/{unique_filename}_training_loss.npy',training_loss)
        np.save(f'statistics/{unique_filename}_training_accuracy.npy',training_accuracy)
        np.save(f'statistics/{unique_filename}_validation_loss.npy',validation_loss)
        np.save(f'statistics/{unique_filename}_validation_accuracy.npy',validation_accuracy)

        return
        
    def train(self):
        training_loss = []
        training_accuracy = []
        validation_loss = []
        validation_accuracy = []
        
        for epoch in range(self.num_epochs):
            # Train the model
            self.model.train()
            
            epoch_loss = []
            epoch_accuracy = []
            for inputs, labels in self.dataloaders['train']:
                print(f'input_size{np.shape(inputs.tensors)},label_size{input.shape(labels.tensors)}')
                # Specify device preference

                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                self.optimizer_ft.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Loss
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()

                # Optimize
                self.optimizer_ft.step()
                
                # Get statistics
                minibatch_loss = loss.item()
                minibatch_accuracy = torch.mean((torch.Tensor.argmax(outputs, axis=1) == labels).type(torch.FloatTensor))
                epoch_loss.append(minibatch_loss)
                epoch_accuracy.append(minibatch_accuracy)   
                inputs = inputs.to(torch.device("cpu"))
                labels = labels.to(torch.device("cpu"))           
            
            # Store training statistics
            training_loss.append(np.mean(epoch_loss))
            training_accuracy.append(np.mean(epoch_accuracy))
                
            # Validate the model
            self.model.eval()
            
            epoch_loss = []
            epoch_accuracy = []
            for inputs, labels in self.dataloaders['val']:
                # Specify device preference
                inputs = inputs.to(device)
                labels = labels.to(device)


                # Forward pass
                outputs = self.model(inputs)
                
                # Loss
                loss = self.criterion(outputs, labels)

                # Get statistics
                minibatch_loss = loss.item()
                minibatch_accuracy = torch.mean((torch.Tensor.argmax(outputs, axis=1) == labels).type(torch.FloatTensor))
                epoch_loss.append(minibatch_loss)
                epoch_accuracy.append(minibatch_accuracy)
                inputs = inputs.to(torch.device("cpu"))
                labels = labels.to(torch.device("cpu"))     
            
            # Store validation statistics
            validation_loss.append(np.mean(epoch_loss))
            validation_accuracy.append(np.mean(epoch_accuracy))
            
            # Decay learning rate
            self.lr_scheduler.step()
            
            # Print statistics
            print('Epoch: %3d \t Training Loss: %.5f \t Training Accuracy: %.5f \t Validation Loss: %.5f \t Validation Accuracy: %.5f' \
                  % (epoch + 1, training_loss[epoch], training_accuracy[epoch], validation_loss[epoch], validation_accuracy[epoch]))

        print('Finished Training')
        self.saveinfo(self.model, training_loss, training_accuracy, validation_loss, validation_accuracy)
        return self.model, training_loss, training_accuracy, validation_loss, validation_accuracy
    
    def test(self):
        self.model.eval()
            
        for inputs, labels in self.dataloaders['test']:
            # Specify device preference
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = self.model(inputs)
            
            # Loss
            loss = self.criterion(outputs, labels)

            # Get statistics
            return loss.item(), torch.mean((torch.Tensor.argmax(outputs, axis=1) == labels).type(torch.FloatTensor))
