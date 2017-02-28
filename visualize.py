import numpy
trainX = numpy.load('tinyX.npy') # this should have shape (26344, 3, 64, 64)
trainY = numpy.load('tinyY.npy')
testX = numpy.load('tinyX_test.npy') # (6600, 3, 64, 64)

# to visualize only
import scipy.misc
scipy.misc.imshow(trainX[0].transpose(2, 1, 0)) # put RGB channels last