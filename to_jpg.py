import numpy as np
import scipy.misc
import cv2

trainX = np.load('tinyX.npy') # this should have shape (26344, 3, 64, 64)
trainY = np.load('tinyY.npy')
trainX = np.flip(trainX, 1)

img = trainX[1].transpose(2, 1, 0)
cv2.imwrite('img_CV2_90.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])

for i in range(trainX.shape[0]):
    img = trainX[i].transpose(2, 1, 0)
    y = trainY[i]
    cv2.imwrite('images/'+str(y)+'/img_CV2_'+str(i)+'.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])