import Neural_Network as NN
import pandas as pd
import numpy as np

train_data = pd.read_csv('train_data.csv')
y_train = train_data['3']
y_train = y_train.as_matrix().reshape(100, 1)
train_data = train_data.ix[:, :'2'].as_matrix()
train_data = train_data.reshape(100, 3)
np.save('train_data', train_data)
np.save('y_train', y_train)

test_data = pd.read_csv('test_data.csv')
y_test = test_data['3']
y_test = y_test.as_matrix().reshape(100, 1)
test_data = test_data.ix[:, :'2'].as_matrix()
test_data = test_data.reshape(100, 3)
np.save('test_data', test_data)
np.save('y_test', y_test)

network = NN.NeuralNetwork(n_inputs=3, n_outputs=2, n_layers=1, n_nodes=3,
                           testX_filename='test_data.npy', trainX_filename='train_data.npy',
                           testY_filename='y_test.npy', trainY_filename='y_train.npy')

print(network.train(0.001, 1.3))

for i in range(test_data.shape[0]):
    print(i, ' ', network.forward_prop(test_data[i, :]))
