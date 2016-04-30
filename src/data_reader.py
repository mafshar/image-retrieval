#!/usr/bin/env python

import scipy.cluster.vq as spvq
import numpy as np
import cv2
import glob
import os
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
np.set_printoptions(threshold='nan')

'''
7.3 thousand images
3k images for training
2k images for validating
1.3k images for testing
'''

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
def get_raw_features(files, output_path, normalize, verbose):
    '''
    Function builds and writes raw features. Reduces the dimensionality of all
    features to 100,000 dimensions, regardless of the dimensionality.

    PARAMETERS:
        files: an array of all the image paths in the directory to extract
            features from
        output_path: string, path to where the data should be written; by
            default it will be OUTPUT_PATH
        normalize: bool value to indicate whether the features should be
                normalized before writing them to disk
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
            img = cv2.resize(img, (350, 400))
        data.append(img.flatten())
        if verbose:
            num += 1
            print "\traw feature for image", num, "took", time.time() - t1,
            print "seconds"
    # end for
    data = PCA(n_components=100000).fit(data).transform(data)
    if normalize:
        data = MinMaxScaler().fit_transform(data)
    print "writing to file..."
    np.save(os.path.join(output_path, "raw_features"), data)
    return

def get_histo_extraction(files, output_path, normalize, verbose):
    '''
    Function builds and writes color histogram features.

    PARAMETERS:
        files: an array of all the image paths in the directory to extract
            features from
        output_path: string, path to where the data should be written; by
            default it will be OUTPUT_PATH
        normalize: bool value to indicate whether the features should be
                normalized before writing them to disk
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
    # end for
    if normalize:
        data = MinMaxScaler().fit_transform(data)
    print "writing to file..."
    np.save(os.path.join(output_path, "histo_features"), data)
    return

def extract_descriptors(files, output_path, verbose, des_type):
    '''
    Function to extract descriptors from images. Writes to a numpy file in the
    output_path.

    PARAMETERS:
        files: an array of all the image paths in the directory to extract
            features from
        output_path: string, path to where the data should be written; by
            default it will be OUTPUT_PATH
        verbose: bool value to indicate whether verbose output is printed to
            console
        des_type: string to indicate what kind of extraction technique should
            be use; extraction techniques are 'sift' and 'surf'; if type not
            recognized, code will return None

    RETURNS:
        If the descriptors already exist in ouput_path, then it will load the
        descriptors and return them. Otherwise, it will return the
        descriptors after creating them.
    '''
    des_filename = des_type.strip().lower() + "_descriptors.npy"
    if os.path.isfile(os.path.join(output_path, des_filename)):
        print "File already exists"
        return np.load(os.path.join(output_path, des_filename))
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
        img = cv2.resize(img, (350, 400))
        if des_type.strip().lower() == "sift":
            extractor = cv2.xfeatures2d.SIFT_create()
        elif des_type.strip().lower() == "surf":
            extractor = cv2.xfeatures2d.SURF_create()
        else:
            print "Descriptor type not recognized."
            return None
        keypoints, des = extractor.detectAndCompute(img, None)
        for item in des:
            descriptors.append(item)
        if verbose:
            num += 1
            if des_type.strip().lower() == "sift":
                print "\tSIFT descriptors for image", num, "took", time.time() - t1,
                print "seconds",
            elif des_type.strip().lower() == "surf":
                print "\tSURF descriptors for image", num, "took", time.time() - t1,
                print "seconds",
            print "with shape", des.shape
    # end for
    print "getting all the descriptors took", time.time() - t0, "seconds"
    print "writing to file..."
    if des_type.strip().lower() == "sift":
        np.save(os.path.join(output_path, "sift_descriptors"), descriptors)
    elif des_type.strip().lower() == "surf":
        np.save(os.path.join(output_path, "surf_descriptors"), descriptors)
    return descriptors

def create_vocabulary(output_path, des_type, k=1024):
    '''
    Function to generate the vocabulary of the image set using the descriptors
        using KMeans Clustering Algorithm -- current implementation uses
        parallelized sklearn.cluster.KMeans()

    PARAMETERS:
        output_path: string, path to where the data should be written; by
            default it will be OUTPUT_PATH
        des_type: string to indicate what kind of extraction technique should
            be use; extraction techniques are 'sift' and 'surf'; if type not
            recognized, code will return None
        k: int value representing the number of clusters to be generated;
            default value is 1024

    RETURNS:
        If the vocabulary already exist in ouput_path, then it will load the
        vocabulary and return the array. Otherwise, it will return the
        vocabulary after creating it.
    '''
    vocab_filename = des_type.strip().lower() + "_vocabulary_" + str(k) + ".npy"
    if os.path.isfile(os.path.join(output_path, vocab_filename)):
        print "File already exists"
        return np.load(os.path.join(output_path, vocab_filename))
    print "starting kmeans clustering"
    t0 = time.time()
    des_filename = des_type.strip().lower() + "_descriptors.npy"
    descriptors = np.load(os.path.join(output_path, des_filename))
    vocab = KMeans(n_clusters=k, max_iter=100, n_jobs=-1).fit(descriptors).cluster_centers_
    print "kmeans took", time.time() - t0, "seconds"
    if des_type.strip().lower() == "sift":
        np.save(os.path.join(output_path, "sift_vocabulary_" + str(k)), vocab)
    elif des_type.strip().lower() == "surf":
        np.save(os.path.join(output_path, "surf_vocabulary_" + str(k)), vocab)
    return vocab

def vector_quantization(images_and_descriptors, vocab):
    '''
    Function to do vector quantization to generate single-row feature vectors
        representing the given images.

    PARAMETERS:
        images_and_descriptors: list of tuples of the form (image_name, descriptors)
            where image_name is a string value and descriptors are the corresponding
            descriptors for that image in a numpy array
        vocab: numpy array of the vocabulary of the image-set

    RETURNS:
        A matrix where each row is a feature vector representing the images in
        the images_and_descriptors list

    '''
    print "starting vector quantization"
    t0 = time.time()
    data = np.zeros((len(images_and_descriptors), len(vocab)))
    for i in range(len(images_and_descriptors)):
        words, distance = spvq.vq(images_and_descriptors[i][1], vocab)
        for word in words:
            data[i][word] += 1
    # end for
    print "vector quantization took", time.time() - t0, "seconds"
    return data


def get_keypoint_features(files, output_path, des_type, weighting, normalize, \
        verbose, vlad=False,):
    '''
    Function that builds and writes keypoint features (SIFT or SURF features)
        Calls:  extract_descriptors(...)
                create_vocabulary(...)
                vector_quantization(...)

    PARAMETERS:
        files: an array of all the image paths in the directory to extract
            features from
        output_path: string, path to where the data should be written; by
            default it will be OUTPUT_PATH
        des_type: string to indicate what kind of extraction technique should
            be use; extraction techniques are 'sift' and 'surf'; if type not
            recognized, code will return None
        weighting: bool value to indicate whether tf-idf weighting should be
            done to the features before writing them to disk
        normalize: bool value to indicate whether the features should be
            normalized before writing them to disk
        verbose: bool value to indicate whether verbose output is printed to
            console
        vlad: bool value to indicate whether it bag-of-visual words
            representation should be used or VLAD feature vectors

    RETURNS:
        If the features already exist in ouput_path, then it will load the
            features and return the feature vectors . Otherwise, it will return
            the features after creating it.
    '''
    filename = des_type + "_features"
    descriptors = extract_descriptors(files, output_path, verbose, des_type)
    vocab = create_vocabulary(output_path, des_type, k=1024)
    images_and_descriptors = get_mapped_descriptors(files, verbose, des_type)
    if vlad:
        print "this is vlad"
        data = []
        filename = "vlad_" + filename
    else:
        data = vector_quantization(images_and_descriptors, vocab)
    if normalize:
        data = MinMaxScaler().fit_transform(data)
    print "writing to file..."
    np.save(os.path.join(output_path, filename), data)

def write_labels(files, output_path, verbose):
    '''
    doc here
    '''
    print "getting the labels"
    t0 = time.time()
    filenames = [f.strip().split("/")[2] for f in files]
    lookup_labels = []
    labels = []
    curr_label = 1
    prev_breed = None
    num = 1
    for f in filenames:
        t1 = time.time()
        curr_breed = f.split("_")[0].strip()
        # print breed
        if not prev_breed:
            prev_breed = curr_breed
            lookup_labels.append((curr_breed, curr_label))
        elif curr_breed != prev_breed:
            curr_label += 1
            prev_breed = curr_breed
            lookup_labels.append((curr_breed, curr_label))
        labels.append(curr_label)
    print "getting all the labels took", time.time() - t0, "seconds"
    print "writing to file..."
    np.save(os.path.join(output_path, "labels"), np.array(labels))
    fh = open(os.path.join(output_path, "lookup_labels.txt"), 'w')
    for item in lookup_labels:
        fh.write(str(item[0]) + " " + str(item[1]) + "\n")
    fh.close()
    return

def build_features(input_path=INPUT_PATH, output_path=OUTPUT_PATH, \
        feature_type="sift", weighting=False, normalize=False, \
            counter=-1, verbose=False):
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




if __name__ == "__main__":
    main()
