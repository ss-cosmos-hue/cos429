from torchvision.models import resnet34,ResNet34_Weights
import torch.nn as nn
import torch
from torchsummary import summary
import torch.optim as optim
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import gc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_img = plt.imread("trashnet/dataset-resized/plastic/plastic1.jpg")
    H,W,_ = np.shape(sample_img)

    classes = ["cardboard","glass","metal","paper","plastic","trash"]
    class_sizes = {"cardboard":403,"glass":501,"metal":410,"paper":594,"plastic":482,"trash":137}
    class_nums =  {"cardboard":0,"glass":1,"metal":2,"paper":3,"plastic":4,"trash":5}

    n_whole = np.sum(list(class_sizes.values()))
    # def gen_dataset():
    X = []
    y = []

    #update X and y
    total_count  = 0
    for categ in classes:
        n_elem = class_sizes[categ]
        for i in range(1,1+n_elem):
            if total_count%100 == 0:
                print(f'total_count = {total_count}')
            # X[total_count] = np.array(Image.open(f'trashnet/dataset-resized/{categ}/{categ}{i}.jpg'))

            X.append(plt.imread(f'trashnet/dataset-resized/{categ}/{categ}{i}.jpg'))
            y.append(class_nums[categ])
            total_count += 1
    print("store in X and y")

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((n_whole,3,H,W))
    print("split")
    train_X,test_X,train_y,test_y = train_test_split(X, y, test_size=0.4, random_state=42)
    del X
    del y
    gc.collect()
    val_X,test_X ,val_y,test_y = train_test_split(test_X, test_y, test_size=0.5, random_state=42)

    print("dataloader train")
    train_X = torch.FloatTensor(train_X)
    train_y = torch.LongTensor(train_y)
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    del train_X
    del train_y
    gc.collect()
    trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=2)
    #not sure we would uuse validationloader or testloader
    print("validation")
    val_X = torch.FloatTensor(val_X)
    val_y = torch.LongTensor(val_y)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_y)
    del val_X
    del val_y
    gc.collect()
    valloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=True, num_workers=2)
    print("test")
    test_X = torch.FloatTensor(test_X)
    test_y = torch.LongTensor(test_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
    del test_X
    del test_y
    gc.collect()
    testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, num_workers=2)

    dataloaders = {'train':trainloader,'val':valloader,'test':testloader}

    ## model initialization 
    model = resnet34()
    model.load_state_dict(torch.load("model_weights.pth"))

    first_layer = list(model.children())[0]
    print(f' first layer: {first_layer}')

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)

    model = model.to(device)

    #summary(model,(3,H,W))
    # 512 x 384 x 3

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    last_layer = list(model.children())[-1]
    print(f'except last layer: {last_layer}')
    for param in last_layer.parameters():
        param.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    score = 0
    outputs = None
    training_losses = []
    validation_losses = []
    training_accs = []
    validation_accs = []
    for epoch in range(1):  # loop over the dataset multiple times
        ##training
        # running_loss = 0.0
        model.train()
        i = 0
        for inputs, labels in dataloaders['train']:
            # get the inputs; data is a list of [inputs, labels]
            print("load to device: input")
            inputs = inputs.to(device)
            print("load to device: labels")
            labels = labels.to(device)
            print("all loaded")
            # zero the parameter gradients
            optimizer_ft.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            print("inferred")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ft.step()

            # print statistics
            # running_loss += loss.item()
            training_losses.append(loss.item())
            score = torch.mean((torch.Tensor.argmax(outputs,axis = 1)==labels).type(torch.FloatTensor))
            training_accs.append(score)
            i += 1

        # ##validation   
        # model.eval()   
        # for inputs, labels in dataloaders['val']:
        #     inputs = inputs.to(device)
        #     labels = labels.to(device)
        #     # Forward Pass
        #     outputs = model(inputs)
        #     #statistics
        #     #loss
        #     validation_loss = criterion(outputs, labels)
        #     validation_losses.append(validation_loss.item())
        #     #accuracy
        #     validation_accs.append(torch.mean((torch.Tensor.argmax(outputs,axis = 1)==labels).type(torch.FloatTensor)))

        if epoch % 1 == 0:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item() / 10:.3f} accuracy: {score}')
            # running_loss = 0.0
        exp_lr_scheduler.step()
    print('Finished Training')

    
    return

if __name__ == "__main__":
    main()
        