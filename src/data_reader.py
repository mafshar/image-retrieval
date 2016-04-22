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
    if whitening:
        descriptors = spvq.whiten(descriptors)
    # else:
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

def samples_writer(files):
    '''
    Creates the feature vectors along with labels for all
    the images and writes to file
    '''
    # looping through the images
    output_fh_features = open(os.path.join(OUTPUT_PATH, "features.txt"), "w")
    output_fh_labels = open(os.path.join(OUTPUT_PATH, "labels.txt"), "w")
    class_number = 0
    prev = None
    curr = None
    i = 0
    for file in files:
        filename = file.split("/")[-1]
        curr = int(filename.split(".")[0])
        img = cv2.imread(file)
        if not prev:
            prev = curr
        if curr - prev != 1:
            # we're in a different class!
            class_number += 1
        output_fh_labels.write(str(class_number) + "\n")
        output_fh_features.write("lawl")
        feature_writer(img, output_fh_labels)
        prev = curr
        i += 1
        print i
        # if i > 20:
        #     break
    return


def read_image(path):
    create_output_dir(OUTPUT_PATH)
    files = glob.glob(INPUT_PATH + "*.jpg")
    samples_writer(files)
    return


def main():
    read_image(INPUT_PATH)
    return

if __name__ == "__main__":
    main()
