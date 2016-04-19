
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
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    files = glob.glob(INPUT_PATH + "*.jpg")
    output_fh = open(OUTPUT_PATH + "labels.txt", "w")
    output_fh.write("suhhhh dude")
    return
    class_number = 0
    prev = None
    curr = None
    counter = 0
    # looping through the images
    for file in files:
        filename = file.split("/")[-1]
        curr = int(filename.split(".")[0])
        if not prev:
            prev = curr
        if curr - prev != 1:
            # we're in a different class!
            class_number += 1
        output_fh.write(class_number)
        output_fh.write("suhhhh dude")
        prev = curr
        if counter > 20:
            break
        counter += 1
    return

if __name__ == "__main__":
    main()
