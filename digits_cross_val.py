from sklearn import datasets
import numpy as np
import pandas as pd
import Neural_Network_Opt as NN
from decimal import *

#overfitting measured by validation loss

# def n_nodes_cross_val():
#
# def n_layers_cross_val():
#
# def alpha_cross_val():


if __name__ == "__main__":

    getcontext().prec = 2

    digits = datasets.load_digits()
    digit_data = digits.data
    digit_labels = digits.target

    n_samples = digit_data.shape[0] #samples = 1797
    n_features = digit_data.shape[1]

    batch_size = 100
    epoch_size = 500
    tolerance = 4
    alpha = 0.01
    decay = 0.9


    # 1 to 4 hidden layers
    for layers in range(1, 5):
        for nodes in range(16, int(n_features/layers)+1, 4):

            # construct neural network based on current hyperparameters
            network = NN.NeuralNetwork(n_nodes=nodes, n_layers=layers, n_inputs=n_features, n_outputs=10,
                                       batch_size=batch_size, test_size=197, fromFile=False, data_array=digit_data, label_array=digit_labels)


            network.train(alpha=alpha, epoch_size=epoch_size, n_train_layers=layers)
            val_loss = network.compute_validation_loss()
            train_loss = network.compute_training_loss()
            tol_count = 0
            epoch = 1

            last_val_loss = val_loss
            while (tol_count < tolerance):
                epoch+=1
                network.reset_state()
                network.train(alpha=alpha*decay, epoch_size=epoch_size, n_train_layers=layers)
                val_loss = network.compute_validation_loss()
                train_loss = network.compute_training_loss()
                print(epoch, train_loss, val_loss)
                if val_loss > last_val_loss:
                    tol_count+=1

                last_val_loss = val_loss

            train_accuracy = network.predict_training()
            val_accuracy = network.validate()
            print('layers: ', layers, 'nodes: ', nodes)
            print('epochs: ', epoch, 'train_acc: ', train_accuracy, 'val_acc: ', val_accuracy)