import numpy as np
import scipy.misc
import cv2

testX = np.load('tinyX_test.npy')
testX = np.flip(testX, 1)

for i in range(testX.shape[0]):
    img = testX[i].transpose(2, 1, 0)
    cv2.imwrite('images/img_CV2_'+str(i)+'.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])