import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1/(1+np.exp(-x))


class NetworkLayer:

    def __init__(self, weight_matrix):
        """
        constructs a layer for a neural network
        :param weight_matrix: matrix of weights used to compute the output of this layer from an input vector
        """
        self.weight_matrix = weight_matrix
        self.n_inputs = weight_matrix.shape[1]
        self.n_outputs = weight_matrix.shape[0]

    def compute_output_vector(self, input_vector):
        return sigmoid(self.weight_matrix.dot(input_vector))


class NeuralNetwork:

    def __init__(self, n_nodes, n_layers, n_inputs, n_outputs, batch_size, test_size,
                 fromFile, trainX_filename = None, trainY_filename = None, data_array = None, label_array = None):
        """
        constructs a neural network made up of several network_layer instances
        :param n_nodes: number of nodes per hidden layer
        :param n_layers: number of hidden layers + output layer
        :param n_inputs: number of input nodes
        :param n_outputs: number of output nodes
        :param batch_size: batch size for vectorized backpropagation
        :param test_size: number of samples in validation set
        :param fromFile: set to True to load data from npy file
        :param trainX_filename: filename of training examples
        :param trainY_filename: filename of training labels
        :param data_array: numpy array of data set
        :param label_array: numpy array of data set labels
        """
        self.n_layers = n_layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_nodes = n_nodes
        self.test_size = test_size

        if fromFile == True:
            self.trainX = np.load(trainX_filename)
            self.trainY = np.load(trainY_filename)

        else:
            self.trainX = data_array
            self.trainY = label_array

        self.trainX = self.trainX.reshape(self.trainX.shape[0], self.n_inputs)
        self.training_set, self.test_set, self.training_outputs, self.test_set_outputs = \
            train_test_split(self.trainX, self.trainY, test_size=self.test_size)

        # network layers' output values in matrix form
        self.node_values = []

        # list of network_layer objects
        self.layer_list = []

        # batch of training examples
        self.batch_size = batch_size
        self.n_batches = int(self.training_set.shape[0]/self.batch_size)
        self.at_batch = 0

        if n_layers == 1:
            weight_mat = np.random.rand(n_outputs, n_inputs)
            self.layer_list.append(NetworkLayer(weight_mat))

        else:
            # construct the first hidden layer's weight matrix using a uniform distribution
            weight_mat = np.random.rand(n_nodes, n_inputs) - 0.5*np.ones((n_nodes, n_inputs))

            # append the first input layer to the layer_list
            self.layer_list.append(NetworkLayer(weight_mat))

            # construct the remaining hidden layers
            for i in range(1, n_layers-1):
                weight_mat = np.random.rand(n_nodes, n_nodes) - 0.5*np.ones((n_nodes, n_nodes))
                self.layer_list.append(NetworkLayer(weight_mat))

            # construct the output layer
            weight_mat = np.random.rand(n_outputs, n_nodes) - 0.5*np.ones((n_outputs, n_nodes))
            self.layer_list.append(NetworkLayer(weight_mat))


    def reset_state(self):
        # resplit the data
        self.cross_val_split()

        # randomize layer weights
        if self.n_layers == 1:
            weight_mat = np.random.rand(self.n_outputs, self.n_inputs)
            self.layer_list[0] = NetworkLayer(weight_mat)

        else:
            # construct the first hidden layer's weight matrix using a uniform distribution
            weight_mat = np.random.rand(self.n_nodes, self.n_inputs) - 0.5*np.ones((self.n_nodes, self.n_inputs))

            # append the first input layer to the layer_list
            self.layer_list[0] = NetworkLayer(weight_mat)

            # construct the remaining hidden layers
            for i in range(1, self.n_layers-1):
                weight_mat = np.random.rand(self.n_nodes, self.n_nodes) - 0.5*np.ones((self.n_nodes, self.n_nodes))
                self.layer_list[i] = NetworkLayer(weight_mat)

            # construct the output layer
            weight_mat = np.random.rand(self.n_outputs, self.n_nodes) - 0.5*np.ones((self.n_outputs, self.n_nodes))
            self.layer_list[self.n_layers-1]=NetworkLayer(weight_mat)


    def create_batch(self):
        self.at_batch += 1
        start_lim = (self.at_batch-1) * self.batch_size

        end_lim = self.at_batch*self.batch_size
        self.batch = self.training_set[start_lim:end_lim].T

        # create the output vectors as a batch, it is sparse and uses memory at the cost of speed
        self.output_vectors_mat = np.zeros((self.n_outputs, self.batch_size))
        for ex in range(start_lim, end_lim):
            self.output_vectors_mat[self.training_outputs[ex], ex-start_lim] = 1

    def cross_val_split(self):
        self.training_set, self.test_set, self.training_outputs, self.test_set_outputs = \
            train_test_split(self.trainX, self.trainY, test_size=self.test_size)

    def forward_prop_batch(self):
        self.node_values.append(self.batch)
        input_vec = self.batch
        for layer in self.layer_list:
            # compute the output in vector form for each layer
            layer_output = layer.compute_output_vector(input_vec)
            # the output is the new input vector for the next layer
            input_vec = layer_output
            # append the outputs into node_values to use for backpropagation
            self.node_values.append(layer_output)

        return layer_output

    def forward_prop(self, input_vec):
        input_vec = input_vec.reshape(self.n_inputs, 1)

        # list of network layers' output values in vector form
        self.node_values = []
        self.node_values.append(input_vec)
        for layer in self.layer_list:
            # compute the output in vector form for each layer
            layer_output = layer.compute_output_vector(input_vec)
            # the output is the new input vector for the next layer
            input_vec = layer_output
            # append the outputs into node_values to use for backpropagation
            self.node_values.append(layer_output)

        return layer_output.reshape(self.n_outputs, 1)

    def compute_training_loss(self):
        J = np.zeros((self.n_outputs, 1))
        for i in range(len(self.training_set)):

            # construct output vector
            y = np.zeros((self.n_outputs, 1))
            y[self.training_outputs[i]] = 1

            h = self.forward_prop(self.training_set[i])
            ones = np.ones((self.n_outputs, 1))
            # use numpy broadcasting to compute the cost in vector form
            J += y*np.log(h) + (ones-y)*np.log(ones-h)

        # after looping through training set add elements of the cost vector
        J = (-1)*np.sum(J)/len(self.training_set)
        return J

    def compute_validation_loss(self):
        J = np.zeros((self.n_outputs, 1))
        for i in range(len(self.test_set)):

            # construct output vector
            y = np.zeros((self.n_outputs, 1))
            y[self.test_set_outputs[i]] = 1

            h = self.forward_prop(self.test_set[i])
            ones = np.ones((self.n_outputs, 1))
            # use numpy broadcasting to compute the cost in vector form
            J += y*np.log(h) + (ones-y)*np.log(ones-h)

        # after looping through training set add elements of the cost vector
        J = (-1)*np.sum(J)/len(self.test_set)
        return J

    def train(self, alpha, epoch_size, n_train_layers):
        # J = self.compute_cost()
        i = 0
        while i < epoch_size:
            print('iteration', i)
            i += 1
            # compute the gradient and update weights in backpropagation
            self.backpropagation(alpha, n_train_layers)
            self.at_batch = 0
            # J = self.compute_cost()


    def backpropagation(self, alpha, n_train_layers):
        # list of delta matrices for each layer
        delta_mat_lst = [0 for i in range(self.n_layers)]

        layer_idx = 0
        for layer in range(self.n_layers - n_train_layers, self.n_layers):
            # get number of rows and cols
            rows = self.layer_list[layer].n_outputs
            cols = self.layer_list[layer].n_inputs

            delta_mat_lst[layer_idx]=(np.zeros((rows, cols)))
            layer_idx+=1

        # loop through training examples, batch by batch
        for i in range(self.n_batches):
            self.node_values = []
            self.create_batch()

            # forward propagate to compute node values
            self.forward_prop_batch()

            delta_lst = [0 for i in range(self.n_layers)]

            # compute delta value for the output layer
            delta_lst[-1] = self.node_values[-1] - self.output_vectors_mat

            a_l = self.node_values[-2]

            # MATH NOTE: sum of outer products for each training example can be expressed as matrix multiplication
            delta_mat_lst[-1] += delta_lst[-1].dot(a_l.T)/len(self.training_outputs)

            # iteratively compute remaining deltas
            for l in reversed(range(self.n_layers - n_train_layers, self.n_layers-1)):

                # get the weight matrix for the next layer
                theta_mat = self.layer_list[l+1].weight_matrix

                # get the output values of this layer
                a_l = self.node_values[l+1]

                # compute delta for the previous level, using numpy broadcasting (*)
                ones = np.ones((a_l.shape[0], self.batch_size))
                delta_lst[l] += (theta_mat.T.dot(delta_lst[l+1]))*a_l*(ones-a_l)

                # update the delta matrix
                delta_mat_lst[l] += delta_lst[l].dot(self.node_values[l].T)/len(self.training_outputs)

        # update the weight matrices based on the gradient components in delta_mat_lst, and step size alpha
        for l in range(self.n_layers - n_train_layers, self.n_layers):
            layer = self.layer_list[l]
            layer.weight_matrix -= alpha*delta_mat_lst[l]

    def validate(self):
        conf_mat = np.array(np.zeros((40, 40)))
        accuracy = 0
        for i in range(self.test_set.shape[0]):
            belief_vec = self.forward_prop(self.test_set[i].reshape(self.n_inputs, 1))
            pred_class = np.argmax(belief_vec, axis=0)
            real_label = self.test_set_outputs[i]
            conf_mat[pred_class, real_label] += 1
            if pred_class == real_label:
                accuracy += 1

        conf_df = pd.DataFrame(conf_mat)
        conf_df.to_csv('conf_matrices/val_conf_matrix.csv')

        return accuracy/self.test_set.shape[0]

    def predict_training(self):
        conf_mat = np.array(np.zeros((40, 40)))
        accuracy = 0
        for i in range(self.training_set.shape[0]):
            belief_vec = self.forward_prop(self.training_set[i].reshape(self.n_inputs, 1))
            pred_class = np.argmax(belief_vec, axis=0)
            real_label = self.training_outputs[i]
            conf_mat[pred_class, real_label] += 1
            if pred_class == real_label:
                accuracy += 1

        conf_df = pd.DataFrame(conf_mat)
        conf_df.to_csv('conf_matrices/train_conf_matrix.csv')

        return accuracy/self.training_set.shape[0]

    def save_network(self, name):
        idx = 0
        for layer in self.layer_list:
            np.save(name+str(idx), layer.weight_matrix)
            idx += 1

    def load_network(self, name):
        for l in range(self.n_layers):
            layer_mat = np.load(name+str(l)+'.npy')
            self.layer_list[l].weight_matrix = layer_mat

    def load_network_layer(self, layer_name, layer_num):
        layer_mat = np.load(layer_name+'.npy')
        self.layer_list[layer_num].weight_matrix = layer_mat
        print(self.layer_list[layer_num].weight_matrix.shape)
