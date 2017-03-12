import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

trainX = np.load('tinyX.npy')  # this should have shape (26344, 3, 64, 64)
trainY = np.load('tinyY.npy')
testX = np.load('tinyX_test.npy')

print("Flattening...")
trainX_flattened = np.zeros((26344, 12288))
i = 0
for i in range(len(trainX)):
    a = trainX[i].flatten()
    trainX_flattened[i] = a
testX_flattened = np.zeros((len(testX), 12288))
j = 0
for i in range(len(testX)):
    b = testX[j].flatten()
    testX_flattened[j] = b
# to visualize only
# plt.imshow(trainX[18015].transpose(2, 1, 0))
# plt.show()


logistic = linear_model.LogisticRegression(verbose=1)

print("Fitting data...")
logistic.fit(trainX_flattened, trainY)

print("Prediction testset...")
pred = logistic.predict(testX_flattened)

pred_df = pd.DataFrame(pred)
pred_df.to_csv("predictions.csv", index_label="id")




# import cv2
#
# print('Read Image')
#
# img = cv2.imread('home.jpg')
# print('Convert to grayscale')
#
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# print('Create SIFT object')
#
# sift = cv2.xfeatures2d.SIFT_create()
# kp = sift.detect(gray, None)
# print('Draw Keypoints')
#
# img = cv2.drawKeypoints(gray, kp)
# print('Write picture with Keypoints to image')
#
# cv2.imwrite('sift_keypoints.jpg', img)
