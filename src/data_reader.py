#!/usr/bin/env python

import scipy.cluster.vq as spvq
import numpy as np
import cv2
import glob
import os
import time
from sklearn.preprocessing import MinMaxScaler
np.set_printoptions(threshold='nan')


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

def surf_extraction(files, output_fh_features, output_fh_features, \
        weighting, normalize, counter, vlad=False):
    '''

    '''
    descriptors = []
    images_and_descriptors = [] ## matrix of (image_path, descriptors)
    for file in files:
        t0 = time.time()
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (1024, 768))
        sift = cv2.xfeatures2d.SURF_create()
        keypoints, des = sift.detectAndCompute(img, None)
        images_and_descriptors.append((file, des))
        for item in des:
            descriptors.append(item)
        print time.time() - t0
        print des.shape
        counter -= 1
        if counter == 0:
            break
        ## now do k-means clustering:
        t0 = time.time()
        k = 1024
        vocab, variance = spvq.kmeans(descriptors, k, 1)
        ## vector quantization:
        data = np.zeros((len(images_and_descriptors), k))
        for i in range(len(images_and_descriptors)):
            words, distance = spvq.vq(images_and_descriptors[i][1], vocab)
            for word in words:
                data[i][word] += 1
        if normalize:
            data = MinMaxScaler().fit_transform(data)
    return


def build_features(feature_type="sift", weighting=False, normalize=False. counter=-1):
    '''
    proper documentation here
    '''
    files = glob.glob(os.path.join(INPUT_PATH, "*.jpg"))
    output_fh_features = open(os.path.join(OUTPUT_PATH, "features.txt"), "w")
    output_fh_labels = open(os.path.join(OUTPUT_PATH, "labels.txt"), "w")

    if feature_type.lower() == "raw":
        return img.flatten()
    elif feature_type.lower() == "histo":
        return cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], \
            [0, 256, 0, 256, 0, 256]).flatten()
    elif feature_type.lower() == "color-mean":
        mean, stdv = cv2.meanStdDev(img)
        return np.concatenate([means, stdv]).flatten()
    elif feature_type.lower() == "sift":
        print "sifty"
    elif feature_type.lower() == "sift-vlad":
        print "sifty-vlad"
    elif feature_type.lower() == "surf":
        print "surfy"
        surf_extraction(files, output_fh_features, output_fh_features, \
                weighting, normalize, counter, vlad=False)
    elif feature_type.lower() == "surf-vlad":
        print "surfy-vlad"
    elif feature_type.lower() == "alexnet":
        print "alexnet"
    else:
        print "Feature type not recognized"

    return


## all images are in grayscale
def main():
    # create_output_dir(OUTPUT_PATH)




    return


if __name__ == "__main__":
    main()
