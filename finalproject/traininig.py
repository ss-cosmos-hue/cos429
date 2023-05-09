import model1
from data import DataCollector
# import matplotlib.pyplot as plt
import numpy as np
# from torchinfo import summary
import torch

height = 512
width = 384
classes = {"cardboard":(0, 403),
        "glass":(1, 501),
        "metal":(2, 410),
        "paper":(3, 594),
        "plastic":(4, 482)}

classes_original = {"cardboard":(0, 34),
        "glass":(1, 20),
        "metal":(2, 26),
        "paper":(3, 42),
        "plastic":(4, 33)}

        # "trash":(5, 137)}
# numlimit = 200
# classes = {"cardboard":(0, numlimit),
#         "glass":(1, numlimit),
#         "metal":(2, numlimit),
#         "paper":(3, numlimit),
#         "plastic":(4, numlimit),
#         "trash":(5, min(137,numlimit))}

skin_tones = {0:(41, 23, 9),#darker
              1:(95, 51, 16),
              2:(127, 68, 34),
              3:(178, 102, 68),
              4:(115, 63, 23),
              5:(147, 95, 55),
              6:(173, 138, 96),
              7:(207, 150, 95),
              8:(187, 101, 54),
              9:(212, 158, 122),
              11:(242, 194, 128),
              12:(236, 192, 145),
              13:(249, 212, 160),
              14:(248, 217, 152),
              15:(253, 231, 173),
              16:(254, 227, 197)}#paler

collector1 = DataCollector(height, width, classes)
datasets1 = collector1.collect('trashnet/dataset-resized',resnet_label = 34, val_size=0.2, test_size=0.2)
collector2 = DataCollector(height, width, classes,mask_opts = {"mask_size":20,"skintone_label":15,})
datasets2 = collector2.collect('trashnet/dataset-resized',resnet_label=34, val_size=0.2, test_size=0.2)
collector3 = DataCollector(height, width, classes_original)
datasets3 = collector3.collect('original_trashnet/original_dataset',resnet_label = 34, val_size=0.25, test_size=0.5,class_balanced=False)

unique_filename_model1_unfreeze_1 = "seqno5_model1_train_final_layer"

model = model1.Model(
    height=height,
    width=width,
    num_classes=len(classes),
    lr=0.001,
    bs = 64,
    rho = 0.9,
    gamma_step = 0.5,
    gamma = 0.9,
    num_active_layers = 5,#
    free_all = False,
    if_replace = True,
    workers = 1,
    epochs = 15,
    mask = False,
    unique_filename = unique_filename_model1_unfreeze_1,
    model_type = 1,
    model_weights_ref_path = "ResNet34_Weights.pth"
    )

train_data_loader = model.construct_data(datasets1)

_, avg_training_loss, avg_training_accuracy, validation_loss, validation_accuracy = model.train()

test_loss, test_accuracy = model.test()
print("testloss",test_loss,"testacc" ,test_accuracy)
print("model1_test_acc_against1",model.test_against_newdata(datasets1['test'])
print("model1_test_acc_against2",model.test_against_newdata(datasets2['test'])
print("model1_test_acc_against3",model.test_against_newdata(datasets3['test'])
      
model2 = model1.Model(
    height=height,
    width=width,
    num_classes=len(classes),
    lr=0.001,
    bs = 64,
    rho = 0.9,
    gamma_step = 0.5,
    gamma = 0.9,
    num_active_layers = 5,
    free_all = False,
    if_replace = True,
    workers = 1,
    epochs = 15,
    mask = True,
    unique_filename = "seqno5_model2",
    model_type = 2,
    model_weights_ref_path = f'models/{unique_filename_model1_unfreeze_1}/model.pth'
    )
      
model2.construct_data(datasets2)

_, avg_training_loss, avg_training_accuracy, validation_loss, validation_accuracy = model2.train()

test_loss, test_accuracy = model2.test()
print("testloss",test_loss,"testacc" ,test_accuracy)

print("model2_test_acc_against1",model2.test_against_newdata(datasets1['test']))
print("model2_test_acc_against2",model2.test_against_newdata(datasets2['test']))
print("model2_test_acc_against3",model2.test_against_newdata(datasets3['test']))
      
      
unique_filename_model2 = "seqno5_model2"
model3 = model1.Model(
    height=height,
    width=width,
    num_classes=len(classes),
    lr=0.001,
    bs = 64,
    rho = 0.9,
    gamma_step = 0.5,
    gamma = 0.9,
    num_active_layers = 5,
    free_all = False,
    if_replace = True,
    workers = 1,
    epochs = 8,
    mask = True,
    unique_filename = "seqno5_model3",
    model_type = 2,
    model_weights_ref_path = f'models/{unique_filename_model2}/model.pth'
    )

model3.construct_data(datasets3)
      
      
      
_, avg_training_loss, avg_training_accuracy, validation_loss, validation_accuracy = model3.train()
      
print("testloss",test_loss,"testacc" ,test_accuracy)

print("model3_test_acc_against1",model3.test_against_newdata(datasets1['test'])
print("model3_test_acc_against2",model3.test_against_newdata(datasets2['test'])
print("model3_test_acc_against3",model3.test_against_newdata(datasets3['test'])