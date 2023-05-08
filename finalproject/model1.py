import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import os
# import gc

device = torch.device('cuda')
device_cpu = torch.device('cpu')


class Model:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.dataloaders = {}
        self.height = kwargs.get('height')
        self.width = kwargs.get('width')
        self.num_classes = kwargs.get('num_classes')
        self.learning_rate = kwargs.get('lr')
        self.batch_size = kwargs.get('bs')
        self.momentum = kwargs.get('rho')
        self.decay_step = kwargs.get('gamma_step')
        self.decay_proportion = kwargs.get('gamma')
        self.free_all = kwargs.get('free_all')
        self.num_active_layers = kwargs.get('num_active_layers')
        self.replace = kwargs.get('if_replace')
        self.num_workers = kwargs.get('workers')
        self.num_epochs = kwargs.get('epochs')
        self.mask = kwargs.get('mask')
        self.unique_filename = kwargs.get('unique_filename')
        self.filepath = 'models/%s' % self.unique_filename
        self.model_type = kwargs.get('model_type')
        self.model_weights_ref_path = kwargs.get('model_weights_ref_path')
        self.create_directory()
        self.configure_model()
    
    ### Create directory for the model
    def create_directory(self):
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        
        # Save keyword arguments
        with open(f'{self.filepath}/arguments.txt',"w+") as file:
            for key, value in self.kwargs.items():
                file.write("{}: {}\n".format(key, value))
    
    ### Edit library model to match the problem
    def configure_model(self):
        # Use library model
        self.model = resnet34()
        # self.model = resnet18()
        
        if self.model_type == 1:
            self.model.load_state_dict(torch.load(self.model_weights_ref_path))
        
        # Make linear layer with number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)
        
        if self.model_type == 2:
            self.model.load_state_dict(torch.load(self.model_weights_ref_path))
        
        
        # Specify device preference
        self.model = self.model.to(device)
        
        # Freeze parameters except for given number of active layers
        if self.free_all:
            for param in self.model.parameters():
                param.requires_grad = True
        else:    
            for param in self.model.parameters():
                param.requires_grad = False
            for i in range(self.num_active_layers):    
                active_layer = list(self.model.children())[-(i+1)]
                for param in active_layer.parameters():
                    param.requires_grad = True
                    
        
    ### Construct the dataloaders using dataset
    def construct_data(self, datasets):
        # Generate data loaders
        self.dataloaders['train'] = torch.utils.data.DataLoader(dataset=datasets['train'], batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.dataloaders['val'] = torch.utils.data.DataLoader(dataset=datasets['val'], batch_size = len(datasets['val']), shuffle = False)
        self.dataloaders['test'] = torch.utils.data.DataLoader(dataset=datasets['test'], batch_size = len(datasets['test']), shuffle = False)

        # Define loss optimization
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_ft = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer_ft, step_size=self.decay_step, gamma=self.decay_proportion)
        
        return self.dataloaders['train'] 
    
    ### Train the model
    def train(self):
        print("HELLO")
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
                # Specify device preference
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print("HEY")
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
               
                # inputs = inputs.to(device_cpu)
                # labels = labels.to(device_cpu)
                # del inputs
                # del labels
                # torch.cuda.empty_cache()
                # gc.collect()
            self.lr_scheduler.step()
                
            
            
            # Store training statistics
            training_loss.append(np.mean(epoch_loss))
            training_accuracy.append(np.mean(epoch_accuracy))
                
            # Validate the model
            self.model.eval()
            
            epoch_loss = []
            epoch_accuracy = []
            count_numnum = 0
               
            for  inputs, labels in self.dataloaders['val']:
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
                
                # inputs = inputs.to(device_cpu)
                # labels = labels.to(device_cpu)
                # del inputs
                # del labels
                # torch.cuda.empty_cache()
                # gc.collect()
            
            # Store validation statistics
            validation_loss.append(np.mean(epoch_loss))
            validation_accuracy.append(np.mean(epoch_accuracy))
            
            # Decay learning rate


            
            # Print statistics
            print('Epoch: %3d \t Training Loss: %.5f \t Training Accuracy: %.5f \t Validation Loss: %.5f \t Validation Accuracy: %.5f' \
                  % (epoch + 1, training_loss[epoch], training_accuracy[epoch], validation_loss[epoch], validation_accuracy[epoch]))

        print('Finished Training')
        
        # Save model
        torch.save(self.model.state_dict(), f'{self.filepath}/model.pth')
        
        # Save statistics
        np.save(f'{self.filepath}/training_loss.npy', training_loss)
        np.save(f'{self.filepath}/training_accuracy.npy', training_accuracy)
        np.save(f'{self.filepath}/validation_loss.npy', validation_loss)
        np.save(f'{self.filepath}/validation_accuracy.npy', validation_accuracy)
                
        return self.model, training_loss, training_accuracy, validation_loss, validation_accuracy
    
    ### Test the model
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
            test_loss = loss.item()
            test_accuracy = torch.mean((torch.Tensor.argmax(outputs, axis=1) == labels).type(torch.FloatTensor))
            print(torch.Tensor.argmax(outputs, axis=1),labels)
            # Save results
            with open(f'{self.filepath}/test_results.txt',"a+") as file:
                file.write(f'Test Loss: %.5f \t Test Accuracy: %.5f' % (test_loss, test_accuracy))
                
            return test_loss, test_accuracy
        
    
    def test_against_newdata(self,dataset):
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size = len(dataset), shuffle = False)
        for inputs, labels in dataloader:
            # Specify device preference
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = self.model(inputs)

            # Loss
            loss = self.criterion(outputs, labels)

            # Get statistics
            test_loss = loss.item()
            test_accuracy = torch.mean((torch.Tensor.argmax(outputs, axis=1) == labels).type(torch.FloatTensor))

            # # Save results
            # with open(f'{self.filepath}/test_results.txt',"a+") as file:
            #     file.write(f'Test Loss: %.5f \t Test Accuracy: %.5f' % (test_loss, test_accuracy))
          
            return test_loss, test_accuracy

