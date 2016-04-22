#!/usr/bin/env python

import scipy.cluster.vq as spvq
import numpy as np
import cv2
import glob
import os
import time

INPUT_PATH = "../data/"
OUTPUT_PATH = "../volume/"

def create_output_dir(path):
    ## creating the output file if it doesn't exist:
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return

# not very useful (will use PCA)
def build_raw_features(img):
    return img.flatten()

# also might not be very useful
def build_color_mean_stdv_features(img):
    mean, stdv = cv2.meanStdDev(img)
    return np.concatenate([means, stdv]).flatten()

def build_flat_histo_features(img):
    return cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], \
        [0, 256, 0, 256, 0, 256]).flatten()

def build_sift_features(gray_img, whitening=False):
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)
    descriptors = spvq.whiten(descriptors)
    return

def build_surf_features():
    return

def feature_writer(img, output_fh_features):
    '''
    builds the feature vector to be used
    '''
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # histo_feature = build_flat_histo_features(img)
    build_sift_features(gray_img)
    return

## all images are in grayscale
def main():
    # create_output_dir(OUTPUT_PATH)
    files = glob.glob(os.path.join(INPUT_PATH + "*.jpg"))
    # looping through the images
    output_fh_features = open(os.path.join(OUTPUT_PATH, "features.txt"), "w")
    output_fh_labels = open(os.path.join(OUTPUT_PATH, "labels.txt"), "w")
    descriptors = []
    images_and_descriptors = {}
    counter = 10
    for file in files:
        # t0 = time.time()
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (1024, 768))
        sift = cv2.xfeatures2d.SURF_create()
        keypoints, des = sift.detectAndCompute(img, None)
        images_and_descriptors[file] = des
        for item in des:
            descriptors.append(item)
        # print time.time() - t0
        # print des.shape
        counter -= 1
        if counter == 0:
            break
    ## whitening helps with kmeans convergence
    descriptors = spvq.whiten(np.array(descriptors))

    ## now do k-means clustering:
    print "starting clustering"
    t0 = time.time()
    k = 1024
    vocab, variance = spvq.kmeans(descriptors, k, 1)
    print time.time() - t0

    ## vector quantization:
    data = []
    feature = np.zeros(k)
    # for img_path,
    #     words, distance = spvq.vq(vocab, file)



    return


if __name__ == "__main__":
    main()
