import numpy as np


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
        return sigmoid(self.weight_matrix.dot(input_vector)).reshape(self.n_outputs, 1)


class NeuralNetwork:

    def __init__(self, n_nodes, n_layers, n_inputs, n_outputs,
                 trainX_filename, trainY_filename, testX_filename, testY_filename):
        """
        constructs a neural network made up of several network_layer instances
        :param n_nodes: number of nodes per hidden layer
        :param n_layers: number of layers, including output layer, not including input layer
        :param n_inputs: number of input nodes
        :param n_outputs: number of output nodes
        :param trainX_filename: filename of training examples
        :param trainY_filename: filename of training labels
        :param testX_filename: filename of testing examples
        :param testY_filename: filename of testing labels
        """
        self.n_layers = n_layers
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_nodes = n_nodes
        self.training_set = np.load(trainX_filename)
        self.training_outputs = np.load(trainY_filename)
        self.test_set = np.load(testX_filename)
        self.test_set_outputs = np.load(testY_filename)

        self.node_values = []

        # list of network_layer objects
        self.layer_list = []

        if n_layers == 1:
            weight_mat = np.random.rand(n_outputs, n_inputs)
            self.layer_list.append(NetworkLayer(weight_mat))

        else:
            # construct the first hidden layer's weight matrix using a uniform distribution
            weight_mat = np.random.rand(n_nodes, n_inputs)

            # append the first input layer to the layer_list
            self.layer_list.append(NetworkLayer(weight_mat))

            # construct the remaining hidden layers
            for i in range(1, n_layers-1):
                weight_mat = np.random.rand(n_nodes, n_nodes)
                self.layer_list.append(NetworkLayer(weight_mat))

            # construct the output layer
            weight_mat = np.random.rand(n_outputs, n_nodes)
            self.layer_list.append(NetworkLayer(weight_mat))

    def forward_prop(self, input_vec):
        input_vec.reshape(self.n_inputs, 1)

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

    def compute_cost(self):
        J = np.zeros((self.n_outputs, 1))
        for i in range(len(self.training_set)):

            # construct output vector
            y = [0 for i in range(self.n_outputs)]
            # set the entry corresponding to the category to 1
            y[self.training_outputs[i, 0]] = 1
            y = np.array(y).reshape(self.n_outputs, 1)

            h = self.forward_prop(self.training_set[i])
            ones = np.ones((self.n_outputs, 1))
            # use numpy broadcasting to compute the cost in vector form
            J += y*np.log(h) + (ones-y)*np.log(ones-h)

        # after looping through training set add elements of the cost vector
        J = (-1)*np.sum(J)/len(self.training_set)
        return J

    def train(self, alpha, tol):
        J = self.compute_cost()
        i = 0
        while J > tol:
            i+=1
            print(i)
            print(J)
            # compute the gradient and update weights in backpropagation
            self.backpropagation(alpha)
            J = self.compute_cost()

        return J

    def backpropagation(self, alpha):
        # list of delta matrices for each layer
        delta_mat_lst = []

        for layer in self.layer_list:
            #get number of rows and cols
            rows = layer.n_outputs
            cols = layer.n_inputs

            delta_mat_lst.append(np.zeros((rows, cols)))

        # loop through training examples
        for i in range(len(self.training_outputs)):

            # forward propagate to compute node values
            self.forward_prop(self.training_set[i])

            delta_lst = [0 for i in range(self.n_layers)]

            # create the output vector
            output_vector = [0 for i in range(self.n_outputs)]
            # set the entry corresponding to the category to 1
            output_vector[self.training_outputs[i, 0]] = 1
            output_vector = np.array(output_vector).reshape(self.n_outputs, 1)

            # compute delta value for the output layer
            delta_lst[-1] = self.node_values[-1] - output_vector
            delta_mat_lst[-1] += delta_lst[-1]*self.node_values[-1]

            # iteratively compute remaining deltas
            for l in reversed(range(self.n_layers - 1)):
                # get the weight matrix for the next layer
                theta_mat = self.layer_list[l+1].weight_matrix

                # get the output values of this layer
                a_l = self.node_values[l+1]

                # compute delta for the previous level, using numpy broadcasting (*)
                ones = np.ones((a_l.shape[0], 1))
                delta_lst[l] = (theta_mat.T.dot(delta_lst[l+1]))*a_l*(ones-a_l)

                # update the delta matrix
                delta_mat_lst[l] = delta_mat_lst[l] + delta_lst[l].dot(self.node_values[l-1].T)

        delta_mat_lst = [mat/len(self.training_outputs) for mat in delta_mat_lst]

        # alpha = self.get_step_size(delta_mat_lst)
        # self.gradient_mat_lst = delta_mat_lst

        #  update the weight matrices based on the gradient components in delta_mat_lst, and step size alpha
        for l in range(self.n_layers):
            layer = self.layer_list[l]
            layer.weight_matrix -= alpha*delta_mat_lst[l]


# #Generate data into csv file
# dim = 4
# I = np.eye(dim)
# np.save('I', I)
# y = np.array(range(dim))
# np.save('y', y)
# NN = NeuralNetwork(2, 2, dim, dim, 'I.npy', 'y.npy', 'I.npy', 'y.npy')
# cost = NN.train(alpha = 0.1, tol = 1)
# print(cost)
# print(NN.layer_list[0].weight_matrix)
# print(NN.layer_list[1].weight_matrix)
#
# for i in range(2):
#     print(NN.forward_prop(I[:, i].reshape(2, 1)))