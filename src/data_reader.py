#!/usr/bin/env python

import scipy.cluster.vq as spvq
import numpy as np
import cv2
import glob
import os
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
np.set_printoptions(threshold='nan')


INPUT_PATH = "../data/"
OUTPUT_PATH = "../volume/"
ERROR_FILES = [ "../data/Abyssinian_34.jpg", \
                "../data/Egyptian_Mau_139.jpg", \
                "../data/Egyptian_Mau_145.jpg",\
                "../data/Egyptian_Mau_167.jpg",\
                "../data/Egyptian_Mau_177.jpg",\
                "../data/Egyptian_Mau_191.jpg"]

def create_dir(path):
    '''
    Function to create a directory in a given path if it doesn't exist.

    PARAMETERS:
        path: string value that is the path where the directory will be created

    RETURNS:
        Nothing.
    '''
    ## creating the output file if it doesn't exist:
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return

## TODO: PCA dimensionality reduction must be done, code NOT runnable
def raw_extraction(files, output_path, normalize, counter, resize, verbose):
    '''
    Function builds and writes raw features. Reduces the dimensionality of all
        features to 100,000 dimensions, regardless of the dimensionality.

    PARAMETERS:
        files: an array of all the image paths in the directory to extract
            features from
        output_fh_features: file-handle for where to write the features
        normalize: bool value to indicate whether the features should be
            normalized before writing them to disk
        counter: int value to determine how many images should be included in
            the feature extraction process; default is -1, indicating all
            features
        verbose: bool value to indicate whether verbose output is printed to
            console

    RETURNS:
        Nothing

    '''
    create_dir(output_path)
    print "obtaining the raw features"
    t0 = time.time()
    data = []
    num = 0
    for file in files:
        if file in ERROR_FILES:
            print "Cannot detect image:", file
            print "Skipping..."
            continue
        t1 = time.time()
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        if resize:
            img = cv2.resize(img, (1024, 768))
        data.append(img.flatten())
        if verbose:
            num += 1
            print "\traw feature for image", num, "took", time.time() - t1,
            print "seconds"
        if counter != -1:
            if counter == 0:
                break
            else:
                counter -= 1
    # end for
    data = PCA(n_components=100000).fit(data).transform(data)
    if normalize:
        data = MinMaxScaler().fit_transform(data)
    print "writing to file..."
    np.save(os.path.join(output_path, "raw_features"), data)
    return

def histo_extraction(files, output_path, normalize, counter, verbose):
    '''
    Function builds and writes color histogram features.

    PARAMETERS:
        files: an array of all the image paths in the directory to extract
            features from
        output_fh_features: file-handle for where to write the features
        normalize: bool value to indicate whether the features should be
            normalized before writing them to disk
        counter: int value to determine how many images should be included in
            the feature extraction process; default is -1, indicating all
            features
        verbose: bool value to indicate whether verbose output is printed to
            console

    RETURNS:
        Nothing

    '''
    create_dir(output_path)
    print "obtaining the histo features"
    t0 = time.time()
    data = []
    num = 0
    for file in files:
        if file in ERROR_FILES:
            print "Cannot detect image:", file
            print "Skipping..."
            continue
        t1 = time.time()
        img = cv2.imread(file)
        data_point = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], \
            [0, 256, 0, 256, 0, 256]).flatten()
        print data_point.shape
        data.append(data_point)
        if verbose:
            num += 1
            print "\tcolor histogram feature for image", num, "took",
            print time.time() - t1, "seconds"
        if counter != -1:
            if counter == 0:
                break
            else:
                counter -= 1
    # end for
    if normalize:
        data = MinMaxScaler().fit_transform(data)
    print "writing to file..."
    np.save(os.path.join(output_path, "histo_features"), data)
    return

def write_descriptors(files, output_path, counter, verbose, des_type="sift"):
    '''
    doc here
    '''
    des_filename = des_type.strip().lower() + "_features.npy"
    if os.path.isfile(os.path.join(output_path, des_filename)):
        print "File already exists"
        return
    descriptors = []
    print "obtaining the descriptors"
    t0 = time.time()
    num = 0
    for file in files:
        if file in ERROR_FILES:
            print "Cannot detect image:", file
            print "Skipping..."
            continue
        t1 = time.time()
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (1024, 768))
        if des_type.strip().lower() == "sift":
            extractor = cv2.xfeatures2d.SIFT_create()
        elif des_type.strip().lower() == "surf":
            extractor = cv2.xfeatures2d.SURF_create()
        else:
            print "Descriptor type not recognized. Exiting..."
            exit(1)
        keypoints, des = extractor.detectAndCompute(img, None)
        for item in des:
            descriptors.append(item)
        if verbose:
            num += 1
            if des_type.strip().lower() == "sift":
                print "\tSIFT descriptors for image", num, "took", time.time() - t1,
                print "seconds"
            elif des_type.strip().lower() == "surf":
                print "\tSURF descriptors for image", num, "took", time.time() - t1,
                print "seconds"
        if counter != -1:
            if counter == 0:
                break
            else:
                counter -= 1
    # end for
    print "getting all the descriptors took", time.time() - t0, "seconds"
    print "writing to file..."
    if des_type.strip().lower() == "sift":
        np.save(os.path.join(output_path, "sift_descriptors"), descriptors)
    elif des_type.strip().lower() == "surf":
        np.save(os.path.join(output_path, "surf_descriptors"), descriptors)
    return

# TODO: vlad feature vectors, labels
def sift_extraction(files, output_path, weighting, normalize, counter, \
        verbose, vlad=False,):
    '''
    Function builds and writes SIFT features.

    PARAMETERS:
        files: an array of all the image paths in the directory to extract
            features from
        output_fh_features: file-handle for where to write the features
        weighting: bool value to indicate whether tf-idf weighting should be
            done to the features before writing them to disk
        normalize: bool value to indicate whether the features should be
            normalized before writing them to disk
        counter: int value to determine how many images should be included in
            the feature extraction process; default is -1, indicating all
            features
        verbose: bool value to indicate whether verbose output is printed to
            console
        vlad: bool value to indicate whether it bag-of-visual words
            representation should be used or VLAD feature vectors

    RETURNS:
        Nothing
    '''
    create_dir(output_path)
    descriptors = []
    images_and_descriptors = [] ## matrix of (image_path, descriptors)
    print "obtaining the descriptors"
    t0 = time.time()
    num = 0
    for file in files:
        if file in ERROR_FILES:
            print "Cannot detect image:", file
            print "Skipping..."
            continue
        t1 = time.time()
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (1024, 768))
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, des = sift.detectAndCompute(img, None)
        images_and_descriptors.append((file, des))
        for item in des:
            descriptors.append(item)
        if verbose:
            num += 1
            print "\tSIFT descriptors for image", num, "took", time.time() - t1,
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
    print "writing to file..."
    if vlad:
        np.save(os.path.join(output_path, "vlad_sift_features"), data)
    else:
        np.save(os.path.join(output_path, "sift_features"), data)
    return

# TODO: vlad feature vectors, labels
def surf_extraction(files, output_path, weighting, normalize, counter, \
        verbose, vlad=False,):
    '''
    Function builds and writes SURF features.

    PARAMETERS:
        files: an array of all the image paths in the directory to extract
            features from
        output_fh_features: file-handle for where to write the features
        weighting: bool value to indicate whether tf-idf weighting should be
            done to the features before writing them to disk
        normalize: bool value to indicate whether the features should be
            normalized before writing them to disk
        counter: int value to determine how many images should be included in
            the feature extraction process; default is -1, indicating all
            features
        verbose: bool value to indicate whether verbose output is printed to
            console
        vlad: bool value to indicate whether it bag-of-visual words
            representation should be used or VLAD feature vectors

    RETURNS:
        Nothing
    '''
    create_dir(output_path)
    descriptors = []
    images_and_descriptors = [] ## matrix of (image_path, descriptors)
    print "obtaining the descriptors"
    t0 = time.time()
    num = 0
    for file in files:
        if file in ERROR_FILES:
            print "Cannot detect image:", file
            print "Skipping..."
            continue
        t1 = time.time()
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        # img = cv2.resize(img, (1024, 768))
        sift = cv2.xfeatures2d.SURF_create()
        keypoints, des = sift.detectAndCompute(img, None)
        images_and_descriptors.append((file, des))
        for item in des:
            descriptors.append(item)
        if verbose:
            num += 1
            print "\tSURF descriptors for image", num, "took", time.time() - t1,
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
    print "writing to file..."
    if vlad:
        np.save(os.path.join(output_path, "vlad_surf_features"), data)
    else:
        np.save(os.path.join(output_path, "surf_features"), data)
    return

def write_labels(files, output_path):
    '''
    doc here
    '''
    return

def build_features(input_path=INPUT_PATH, output_path=OUTPUT_PATH, \
        feature_type="sift", weighting=False, normalize=False, \
            counter=-1, resize=False, verbose=False):
    '''
    Function to extract features.

    PARAMETERS:
        input_path: string, path to where the data is located; by default it
            will be INPUT_PATH
        output_path: string, path to where the data should be written; by
            default it will be OUTPUT_PATH
        feature_type: string, describing the type of feature to be extracted;
            possible values include 'raw', 'histo', 'sift', 'vlad_sift',
            'surf', 'vlad-surf', 'alexnet'; if value is not supported, prints
            error message and returns
        weighting: bool value to indicate whether tf-idf weighting should be
            done to the features before writing them to disk; used for 'sift',
            'surf', 'vlad_sift', 'vlad_surf'
        normalize: bool value to indicate whether the features should be
            normalized before writing them to disk; used by all feature types
        counter: int value to determine how many images should be included in
            the feature extraction process; default is -1, indicating all
            features
        resize: bool value to indicate whether or not the raw images should be
            resized or not; if True, resize value is [1024 x 768], or 786432
            pixels
        verbose: bool value to indicate whether verbose output is printed to
            console

    RETURNS:
        Nothing
    '''
    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.jpg"))
    if feature_type.strip().lower() == "raw":
        raw_extraction(files, output_path, normalize, counter, resize, verbose)
    elif feature_type.strip().lower() == "histo":
        histo_extraction(files, output_path, normalize, counter, verbose)
    elif feature_type.strip().lower() == "sift":
        sift_extraction(files, output_path, weighting, normalize, counter, \
            verbose, vlad=False)
    elif feature_type.strip().lower() == "vlad_sift":
        sift_extraction(files, output_path, weighting, normalize, counter, \
            verbose, vlad=True)
    elif feature_type.strip().lower() == "surf":
        surf_extraction(files, output_path, weighting, normalize, counter, \
            verbose, vlad=False)
    elif feature_type.strip().lower() == "vlad_surf":
        surf_extraction(files, output_path, weighting, normalize, counter, \
            verbose, vlad=True)
    elif feature_type.strip().lower() == "alexnet":
        print "alexnet"
    else:
        print "Feature type not recognized"
    return


## all images are in grayscale
## TODO: ADD A FLAG TO CHECK IF THE DATA IS ALREADY THERE OR NOT
def main():
    create_dir(OUTPUT_PATH) # creates the top-level output directory
    ## this needs to be heavily modified:
    # build_features(output_path=OUTPUT_PATH, feature_type="raw", counter=10,\
    #     verbose=True)
    ## BUILT FEATURE:
    # build_features(output_path=OUTPUT_PATH, feature_type="histo", counter=-1,\
    #     verbose=True)
    # build_features(output_path=OUTPUT_PATH, feature_type="sift", counter=10,\
    #     verbose=True)
    # build_features(output_path=OUTPUT_PATH, feature_type="surf", counter=1500,\
    #     verbose=True)
    # files = glob.glob(os.path.join(INPUT_PATH, "*.jpg"))
    # write_descriptors(files, output_path=OUTPUT_PATH, counter=-1, verbose=True, des_type="surf")
    # tmp = np.load(os.path.join(OUTPUT_PATH, "surf_descriptors.npy"))


if __name__ == "__main__":
    main()
