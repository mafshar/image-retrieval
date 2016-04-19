
import numpy as np
import scipy as sp
import cv2
import glob
import os
import time

INPUT_PATH = "../data/"
OUTPUT_PATH = "../volume/"

def read_image():
    return

def main():
    ## creating the output file if it doesn't exist:
    if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
        try:
            os.makedirs(os.path.dirname(OUTPUT_PATH))
        except OSError as exc: # guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    files = glob.glob(INPUT_PATH + "*.jpg")
    output_fh = open(OUTPUT_PATH + "labels.txt", "w")
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
        output_fh.write(str(class_number) + "\n")
        prev = curr
    return

if __name__ == "__main__":
    main()
