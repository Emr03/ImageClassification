import numpy as np
import pandas as pd
import Neural_Network_Opt as NN


def validate_tol(n_layers, n_inputs, n_nodes, alpha):
    trainX_filename = 'trainX_gray_norm.npy'
    trainY_filename = 'tinyY.npy'

    max_iter = 10000

    columns = ['iterations', 'training accuracy', 'testing_accuracy']
    cv_tol = pd.DataFrame(columns=columns)

    network = NN.NeuralNetwork(n_inputs=n_inputs, n_outputs=40, n_nodes=n_nodes, n_layers=n_layers,
                               trainX_filename=trainX_filename, trainY_filename=trainY_filename)


    network.train(alpha, max_iter)
    training_accuracy = network.predict_training()
    testing_accuracy = network.validate()
    print('training acc', training_accuracy)
    print('testing acc', testing_accuracy)
    network.save_network('firstNN')

    # cv_tol.iloc[0] = pd.DataFrame({'iterations': max_iter,
    #                                  'training accuracy': training_accuracy,
    #                                  'testing accuracy': testing_accuracy}, index=[0])


    idx = 1
    prev_accuracy = testing_accuracy
    while prev_accuracy < testing_accuracy - 0.05:
        print(idx)
        prev_accuracy = testing_accuracy
        max_iter += 100000
        network.train(alpha, max_iter) # will not reset the weights
        training_accuracy = network.predict_training()
        testing_accuracy = network.validate()
        idx += 1
        # cv_tol.iloc[idx] = pd.DataFrame({'iterations': max_iter,
        #                                  'training accuracy': training_accuracy,
        #                                  'testing accuracy': testing_accuracy}, index=[idx])

    network.save_network('firstNN')
    # cv_tol.to_csv('cv_tol.csv')

if __name__ == "__main__":

    n_layers = 2
    n_nodes = int((64*64+1)/2)
    n_inputs = 64*64+1
    alpha = 0.0001
    validate_tol(n_layers, n_inputs, n_nodes, alpha)
