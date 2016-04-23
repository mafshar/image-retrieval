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
    '''
    proper doc here
    '''
    ## creating the output file if it doesn't exist:
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return

# TODO: vlad feature vectors, labels
def surf_extraction(files, output_path, weighting, normalize, counter, \
        verbose, vlad=False,):
    '''
    Function creates a directory in the presently working directory, writes
    features and labels to that directory.

    PARAMETERS:
        files: an array of all the images in the directory to extract features
            from
        output_fh_features: file-handle for where to write the features
        output_fh_labels: file-handle for where to write the labels
        weighting: bool value to indicate whether tf-idf weighting should be
            done to the features before writing them to disk
        normalize: bool value to indicate whether the features should be
            normalized before writing them to disk
        counter: bool value to indicate whether verbose output is printed to
            console
        vlad: bool value to indicate whether it bag-of-visual words
            representation should be used or VLAD feature vectors

    RETURNS:
        Nothing
    '''
    descriptors = []
    images_and_descriptors = [] ## matrix of (image_path, descriptors)
    print "obtaining the descriptors"
    t0 = time.time()
    num = 0
    for file in files:
        t1 = time.time()
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (1024, 768))
        sift = cv2.xfeatures2d.SURF_create()
        keypoints, des = sift.detectAndCompute(img, None)
        images_and_descriptors.append((file, des))
        for item in des:
            descriptors.append(item)
        if verbose:
            num += 1
            print "\tdescriptor for image", num, "took", time.time() - t1,
            print "seconds"
        if counter != -1:
            if counter == 0:
                break
            else:
                counter -= 1
    # end for
    print "getting all the descriptors took", time.time() - t0, "seconds"
    ## now do k-means clustering:
    print "starting kmeans clustering"
    t2 = time.time()
    k = 1024
    vocab, variance = spvq.kmeans(descriptors, k, 1)
    print "kmeans took", time.time() - t2, "seconds"
    ## vector quantization:
    print "starting vector quantization"
    t3 = time.time()
    data = np.zeros((len(images_and_descriptors), k))
    for i in range(len(images_and_descriptors)):
        words, distance = spvq.vq(images_and_descriptors[i][1], vocab)
        for word in words:
            data[i][word] += 1
    # end for
    print "vector quantization took", time.time() - t3, "seconds"
    if normalize:
        data = MinMaxScaler().fit_transform(data)

    return


def build_features(input_path=INPUT_PATH, output_path=OUTPUT_PATH, \
        feature_type="sift", weighting=False, normalize=False, counter=-1, verbose=False):
    '''
    proper documentation here
    '''
    files = glob.glob(os.path.join(input_path, "*.jpg"))
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
        surf_extraction(files, output_path, weighting, normalize, counter, \
            verbose, vlad=False,)
    elif feature_type.lower() == "surf-vlad":
        print "surfy-vlad"
    elif feature_type.lower() == "alexnet":
        print "alexnet"
    else:
        print "Feature type not recognized"

    return

## all images are in grayscale
def main():
    create_output_dir(OUTPUT_PATH)
    build_features(feature_type="surf", counter=10, verbose=True)


    return


if __name__ == "__main__":
    main()
