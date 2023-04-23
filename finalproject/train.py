from data import collect_data, gen_dataset
from model1 import Model

X, y = collect_data()
datasets = gen_dataset(X, y, val_size = 0.2, test_size = 0.2)

H = 512
W = 384

model1 = Model(
    lr=0.001,
    bs = 64,
    rho = 0.9,
    gamma_step = 0.5,
    gamma = 0.9,
    n_active_layers = 1,
    if_replace = True,
    workers = 2,
    epochs = 2,
    mask = False,
    unique_filename = "seqno1"
    )

model1.construct_data(datasets)

_, avg_training_loss, avg_training_accuracy, validation_loss, validation_accuracy = model1.train()

test_loss, test_acc = model1.test()