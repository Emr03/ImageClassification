import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray

def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.02, edgeThreshold=20, sigma=1)
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc

def show_sift_features(gray_img, color_img, kp):
    return plt.imshow(cv2.drawKeypoints(gray_img, kp, color_img.copy()))

trainX = np.load('data/tinyX.npy') # this should have shape (26344, 3, 64, 64)
trainY = np.load('data/tinyY.npy')
testX = np.load('data/tinyX_test.npy') # (6600, 3, 64, 64)

feature_data = []
desc_labels = []

# for each image, extract SIFT features
for i in range(trainX.shape[0]):
    img = trainX[i].transpose(2, 1, 0)
    gray_img = to_gray(img)
    kp, desc_array = gen_sift_features(gray_img)
    # for each descriptor, save its 128-dim vector in feature_data, and its label in desc_labels
    if i%100 == 0:
        print('iteration ', i)
    if desc_array is None:
        print(i)
    else:
        for n in range(desc_array.shape[0]):
            feature_data.append(desc_array[n])
            desc_labels.append(trainY[i])

np.save('sift_data', np.array(feature_data))
np.save('sift_data_labels', np.array(desc_labels))

sift_data = np.load('sift_data.npy')
sift_labels = np.load('sift_data_labels.npy')