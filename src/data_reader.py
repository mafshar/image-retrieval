#!/usr/bin/env python

import scipy.cluster.vq as spvq
import scipy.spatial.distance as spsd
import numpy as np
import cv2
import glob
import os
import time
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.neighbors import LSHForest
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.set_printoptions(threshold='nan')

'''
7.3 thousand images
3k images for training
2k images for validating
1.3k images for testing
'''

INPUT_PATH_MULTI = "../data/multi_class/"
INPUT_PATH_BIN = "../data/binary_class/"
OUTPUT_PATH = "../volume/"
OUTPUT_PATH_MULTI = "../volume/multi_class/"
OUTPUT_PATH_BIN = "../volume/binary_class/"
ERROR_FILES = [ "Abyssinian_34.jpg", \
                "Egyptian_Mau_139.jpg", \
                "Egyptian_Mau_145.jpg",\
                "data/Egyptian_Mau_167.jpg",\
                "Egyptian_Mau_177.jpg",\
                "Egyptian_Mau_191.jpg"]

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

def get_raw_features(files, output_path, pca=False, normalize=False, verbose=False):
    '''
    Function builds and writes raw features. Reduces the dimensionality of all
    features to 100,000 dimensions, regardless of the dimensionality.

    PARAMETERS:
        files: an array of all the image paths in the directory to extract
            features from
        output_path: string, path to where the data should be written; by
            default it will be OUTPUT_PATH
        pca: bool value to indicate whether the features dimensionality should
            be reduced to 10,000 instead of 256x256 = 65,536 dimensions
        normalize: bool value to indicate whether the features should be
                normalized before writing them to disk
        verbose: bool value to indicate whether verbose output is printed to
            console

    RETURNS:
        Nothing
    '''
    create_dir(output_path)
    filename = "raw_features"
    print "obtaining the raw features"
    t0 = time.time()
    data = []
    num = 0
    for file in files:
        img_filename = file.strip().split("/")[-1]
        if img_filename in ERROR_FILES:
            print "Cannot detect image:", file
            print "Skipping..."
            continue
        t1 = time.time()
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (256, 256))
        data.append(img.flatten())
        if verbose:
            num += 1
            print "\traw feature for image", num, "took", time.time() - t1,
            print "seconds with shape", img.flatten().shape
    # end for
    print "obtaining the raw features took", time.time() - t0, "seconds"
    if pca:
        t0 = time.time()
        print "starting PCA (without whitening)"
        data = scale(data)
        data = PCA(n_components=10000).fit_transform(data)
        print "PCA took", time.time() - t0, "seconds"
        filename += "_pca"
    if normalize:
        data = MinMaxScaler().fit_transform(data)
        filename += "_normalized"
    print "writing to file..."
    np.save(os.path.join(output_path, filename), data)
    return

def get_histo_extraction(files, output_path, normalize=False, verbose=False):
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
    filename = "histo_features.npy"
    if os.path.isfile(os.path.join(output_path, filename)):
        print "Descriptors file already exists"
        return np.load(os.path.join(output_path, filename))
    filename = "histo_features"
    print "obtaining the histo features"
    t0 = time.time()
    data = []
    num = 0
    for file in files:
        img_filename = file.strip().split("/")[-1]
        if img_filename in ERROR_FILES:
            print "Cannot detect image:", file
            print "Skipping..."
            continue
        t1 = time.time()
        img = cv2.imread(file)
        data_point = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], \
            [0, 256, 0, 256, 0, 256]).flatten()
        data.append(data_point)
        if verbose:
            num += 1
            print "\tcolor histogram feature for image", num, "took",
            print time.time() - t1, "seconds", "with shape", data_point.shape
    # end for
    if normalize:
        data = MinMaxScaler().fit_transform(data)
        filename += "_normalized"
    print "writing to file..."
    np.save(os.path.join(output_path, filename), data)
    return

def extract_descriptors(files, output_path, des_type, verbose=False):
    '''
    Function to extract descriptors from images. Writes to a numpy file in the
    output_path.

    PARAMETERS:
        files: an array of all the image paths in the directory to extract
            features from
        output_path: string, path to where the data should be written; by
            default it will be OUTPUT_PATH
        des_type: string to indicate what kind of extraction technique should
            be use; extraction techniques are 'sift' and 'surf'; if type not
            recognized, code will return None
        verbose: bool value to indicate whether verbose output is printed to
            console

    RETURNS:
        If the descriptors already exist in ouput_path, then it will load the
        descriptors and return them. Otherwise, it will return the
        descriptors after creating them.
    '''
    des_filename = des_type.strip().lower() + "_descriptors.npy"
    if os.path.isfile(os.path.join(output_path, des_filename)):
        print "Descriptors file already exists"
        return np.load(os.path.join(output_path, des_filename))
    descriptors = []
    print "obtaining the descriptors"
    t0 = time.time()
    num = 0
    for file in files:
        img_filename = file.strip().split("/")[-1]
        if img_filename in ERROR_FILES:
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

def get_mapped_descriptors(files, verbose, des_type):
    '''
    Function to extract descriptors from images, mapping which set of descriptors
    belong to which image. Writes to file to avoid a recomputing in a second
    run.

    PARAMETERS:
        files: an array of all the image paths in the directory to extract
            features from
        output_path: string, path to where the data should be written; by
            default it will be OUTPUT_PATH
        des_type: string to indicate what kind of extraction technique should
            be use; extraction techniques are 'sift' and 'surf'; if type not
            recognized, code will return None
        verbose: bool value to indicate whether verbose output is printed to
            console

    RETURNS:
        A list of tuples of the form (image_name, descriptors) where image_name
        is a string value and descriptors are the corresponding descriptors for
        that image in a numpy array.
    '''
    images_and_descriptors = []
    print "obtaining the descriptors"
    num = 0
    t0 = time.time()
    for file in files:
        img_filename = file.strip().split("/")[-1]
        if img_filename in ERROR_FILES:
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
        images_and_descriptors.append((file, des))
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
    return images_and_descriptors

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
        print "Vocab file already exists"
        return np.load(os.path.join(output_path, vocab_filename))
    des_filename = des_type.strip().lower() + "_descriptors.npy"
    descriptors = np.load(os.path.join(output_path, des_filename))
    print "starting kmeans clustering"
    t0 = time.time()
    vocab = KMeans(n_clusters=k, max_iter=100, n_jobs=-1).fit(descriptors).cluster_centers_
    print "kmeans took", time.time() - t0, "seconds"
    if des_type.strip().lower() == "sift":
        np.save(os.path.join(output_path, vocab_filename), vocab)
    elif des_type.strip().lower() == "surf":
        np.save(os.path.join(output_path, vocab_filename), vocab)
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

def vlad_quantization(images_and_descriptors, vocab, verbose=False):
    '''
    Function to do create VLAD (Vector of Locally Aggregated Descriptors)
    to generate single-row feature vector representing the given images.
    Whitening is applied after VLAD feature vectors are created.

    PARAMETERS:
        images_and_descriptors: list of tuples of the form (image_name, descriptors)
            where image_name is a string value and descriptors are the corresponding
            descriptors for that image in a numpy array
        vocab: numpy array of the vocabulary of the image-set

    RETURNS:
        A matrix where each row is a VLAD feature vector representing the
        images in the images_and_descriptors list

    '''
    print "creating VLAD feature vectors"
    t0 = time.time()
    vlad_features = []
    for i in range(len(images_and_descriptors)):
        t1 = time.time()
        for point in images_and_descriptors[i][1]:
            vlad =[0.0] * len(vocab)
            lshf = LSHForest(random_state=42)
            lshf.fit(vocab)
            #get distance using approximate nearest neighbors
            distances, centroid_ndx = lshf.kneighbors(point, n_neighbors=len(vocab))
            for index, centroid in enumerate(centroid_ndx[0]):
                vlad[centroid] = distances[0][index]
            vlad = np.array(vlad)
        if verbose:
            print "vlad feature for image", i + 1, "took", time.time() - t1,
            print "seconds"
        vlad_features.append(vlad)
    vlad_features = np.array(vlad_features)
    data = PCA(whiten=True).fit_transform(vlad_features)
    print "all VLAD feature vectors took", time.time() - t0, "seconds"
    print len(data)
    print len(data)
    print data[0]
    return data

def get_keypoint_features(files, output_path, des_type, weighting=False, \
        normalize=False, verbose=False, vlad=False, k=1024):
    '''
    Function that builds and writes keypoint features (SIFT or SURF features)
        Calls:  extract_descriptors(...)
                create_vocabulary(...) // this is always passed "../volume/"
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
    filename = des_type + "_features_" + str(k)
    descriptors = extract_descriptors(files, output_path, des_type, verbose)
    vocab = create_vocabulary(OUTPUT_PATH, des_type, k)
    images_and_descriptors = get_mapped_descriptors(files, verbose, des_type)
    if vlad:
        data = vlad_quantization(images_and_descriptors, vocab, verbose)
        filename = "vlad_" + filename
    else:
        data = vector_quantization(images_and_descriptors, vocab, verbose)
    if normalize:
        data = MinMaxScaler().fit_transform(data)
        filename += "_normalized"
    print "writing to file..."
    np.save(os.path.join(output_path, filename), data)

def create_multi_labels(input_path=INPUT_PATH_MULTI, \
        output_path=OUTPUT_PATH_MULTI, verbose=False):
    '''
    Function to create the multi-class labels and the dictionary that maps
    a breed name to a label.

    PARAMETERS:
        input_path: path where the data is located - default is
            "../data/multi_class"
        output_path: string, path to where the data should be written; by
            default it will be "../volume/multi_class"
        verbose: bool value to indicate whether verbose output is printed to
            console

    RETURNS:
        Nothing
    '''
    files = glob.glob(os.path.join(input_path, "*.jpg"))
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

def get_animal_files(input_path):
    dog_path = os.path.join(input_path, "dog")
    cat_path = os.path.join(input_path, "cat")
    all_dogs = []
    all_cats = []
    for root, dirs, files in os.walk(dog_path):
        all_dogs.append(root)
    for root, dirs, files in os.walk(cat_path):
        all_cats.append(root)
    all_dogs.pop(0)
    all_cats.pop(0)
    dog_files = []
    cat_files = []
    for path in all_dogs:
        files = glob.glob(os.path.join(path, "*.jpg"))
        dog_files.extend(files)
    for path in all_cats:
        files = glob.glob(os.path.join(path, "*.jpg"))
        cat_files.extend(files)
    return dog_files, cat_files

def create_binary_labels(input_path=INPUT_PATH_BIN, \
        output_path=OUTPUT_PATH_BIN):
    '''
    Function to create the multi-class labels and the dictionary that maps
    a breed name to a label. Assumes dog is +1, cat is -1

    PARAMETERS:
        input_path: path where the data is located - default is
            "../data/binary_class"
        output_path: string, path to where the data should be written; by
            default it will be "../volume/binary_class"

    RETURNS:
        Nothing
    '''
    dog_files, cat_files = get_animal_files(input_path)
    fhd = open(os.path.join(output_path, "dog_labels.txt"), 'w')
    fhc = open(os.path.join(output_path, "cat_labels.txt"), 'w')
    for item in dog_files:
        fhd.write("1\n")
    for item in cat_files:
        fhc.write("-1\n")
    return

def main():
    create_dir(OUTPUT_PATH_MULTI) # creates the top-level output directory
    create_dir(OUTPUT_PATH_BIN) # creates the top-level output directory
    # create_vocabulary(output_path=OUTPUT_PATH, des_type='surf', k=1024)

    all_files = glob.glob(os.path.join(INPUT_PATH_MULTI, "*.jpg"))
    dog_files, cat_files = get_animal_files(INPUT_PATH_BIN)

    ##SIFT
    # get_keypoint_features(files=files, output_path=OUTPUT_PATH, des_type='sift',\
    #     verbose=True)
    # get_keypoint_features(files=dog_files, output_path=OUTPUT_PATH_BIN+"dog", des_type='sift',\
    #     verbose=True)
    # get_keypoint_features(files=cat_files, output_path=OUTPUT_PATH_BIN+"cat", des_type='sift',\
    #     verbose=True)
    ##SURF
    # get_keypoint_features(files=files, output_path=OUTPUT_PATH, des_type='surf',\
    #     verbose=True, k=1000)
    # get_keypoint_features(files=dog_files, output_path=OUTPUT_PATH_BIN+"dog", des_type='surf',\
    #     verbose=True, k=1000)
    # get_keypoint_features(files=cat_files, output_path=OUTPUT_PATH_BIN+"cat", des_type='surf',\
    #     verbose=True, k=1000)
    ##RAW
    # get_raw_features(files=files, output_path=OUTPUT_PATH, pca=True, verbose=True)
    # get_raw_features(files=dog_files, output_path=OUTPUT_PATH_BIN+"dog", pca=True, verbose=True)
    # get_raw_features(files=cat_files, output_path=OUTPUT_PATH_BIN+"cat", pca=True, verbose=True)
    ##HISTO
    # get_histo_extraction(files=files, output_path=OUTPUT_PATH, verbose=True)
    # get_histo_extraction(files=dog_files, output_path=OUTPUT_PATH_BIN+"dog", verbose=True)
    # get_histo_extraction(files=cat_files, output_path=OUTPUT_PATH_BIN+"cat", verbose=True)

    # create_binary_labels(input_path=INPUT_PATH_BIN, output_path=OUTPUT_PATH_BIN)

    get_keypoint_features(files=cat_files, output_path=OUTPUT_PATH_BIN+"cat", des_type='surf',\
        verbose=True, vlad=True, k=256)

    # create_vocabulary(output_path=OUTPUT_PATH, des_type='sift', k=64)
    # create_vocabulary(output_path=OUTPUT_PATH, des_type='surf', k=64)


if __name__ == "__main__":
    main()
