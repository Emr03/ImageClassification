import numpy as np
import pandas as pd
import Neural_Network_Opt as NN
import cv2

def validate_tol(n_layers, n_inputs, n_nodes, alpha):
    trainX_filename = 'sift_data.npy'
    trainY_filename = 'sift_data_labels.npy'

    max_iter = 10000

    columns = ['iterations', 'training accuracy', 'testing_accuracy']
    cv_tol = pd.DataFrame(columns=columns)

    356
    network = NN.NeuralNetwork(n_inputs=n_inputs, n_outputs=40, n_nodes=n_nodes, n_layers=n_layers,
                               batch_size=500, test_size = 344,
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


def test_run(n_layers, n_inputs, n_nodes, alpha):
    trainX_filename = 'tinyX.npy'
    trainY_filename = 'tinyY.npy'
    max_iter = 1000

    network = NN.NeuralNetwork(n_inputs=n_inputs, n_outputs=40, n_nodes=n_nodes, n_layers=n_layers,
                               fromFile=True, batch_size=200, test_size=2433, trainX_filename=trainX_filename, trainY_filename=trainY_filename)

    # network.load_network_layer('thirdNN0', 0)
    # network.load_network_layer('thirdNN1', 1)
    # network.load_network_layer('thirdNN2', 2)
    # network.load_network_layer('thirdNN3', 3)
    # network.load_network_layer('thirdNN4', 4)
    network.train(alpha, max_iter, 1)

    network.save_network('fifthNN')
    training_accuracy = network.predict_training()
    testing_accuracy = network.validate()
    print('training acc', training_accuracy)
    print('testing acc', testing_accuracy)

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

if __name__ == "__main__":
    trainX_filename = 'tinyX.npy'
    trainY_filename = 'tinyY.npy'
    imgs = np.load(trainX_filename)
    gray_imgs = np.zeros((imgs.shape[0], 64, 64))

    # for i in range(imgs.shape[0]):
    #     gray_imgs[i]=to_gray(imgs[i].transpose(2, 1, 0))

    # np.save('gray_imgs', gray_imgs)

    #get sift data
    # sift_data = np.load('sift_data.npy')
    # sift_labels = np.load('sift_data_labels.npy')

    n_layers = 2
    n_nodes = 64
    n_inputs = 3*64*64
    alpha = 0.01
    test_run(n_layers, n_inputs, n_nodes, alpha)
