
import numpy as np
import scipy as sp
import cv2
import glob
import os
import time

INPUT_PATH = "../data/"
OUTPUT_PATH = "../volume/"
TYPES_OF_FEATURES = ["raw", "sift", "surf", "flat-histo"]

def create_output_dir(path):
    ## creating the output file if it doesn't exist:
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc: # guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    return

def build_features():

    return


def read_image(path):
    create_output_dir(OUTPUT_PATH)
    files = glob.glob(INPUT_PATH + "*.jpg")
    output_fh_features = open(OUTPUT_PATH + "features.txt", "w")
    output_fh_label = open(OUTPUT_PATH + "labels.txt", "w")
    class_number = 0
    prev = None
    curr = None
    # looping through the images
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
        build_features()
        prev = curr
    return


def main():
    read_image(INPUT_PATH)
    return

if __name__ == "__main__":
    main()
