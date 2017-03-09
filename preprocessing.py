import cv2
import numpy as np
import scipy.misc
import pandas as pd
import matplotlib.pyplot as plt


def normalize_gray():
    # load grayscale image data
    gray_imgs = np.load('trainX_gray.npy')
    gray_imgs_norm = np.array(np.zeros((gray_imgs.shape[0], 64 * 64 + 1)))

    # for each image, normalize the pixel values and add a bias term
    for i in range(gray_imgs.shape[0]):
        mean = np.mean(gray_imgs[i])
        std = np.std(gray_imgs[i])
        norm_img = (gray_imgs[i] - mean * np.ones((64, 64))) / std
        scipy.misc.imshow(norm_img)
        gray_vec = norm_img.reshape(4096, )
        gray_imgs_norm[i] = np.concatenate((gray_vec, np.array([1]).reshape(1, )), axis=0)

    np.save('trainX_gray_norm', gray_imgs_norm)

def noise_reduction():
    # load grayscale image data
    gray_imgs = np.load('trainX_gray.npy')
    gray_imgs_filtered = np.array(np.zeros((gray_imgs.shape[0], 64 * 64 + 1)))

    kernel = np.ones((4, 4), np.float32) / 16
    for i in range()
    dst = cv2.filter2D(img, -1, kernel)



def canny_edge_detector():
