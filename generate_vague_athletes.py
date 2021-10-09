import os
import cv2
import sys
import numpy as np

from tqdm import trange, tqdm

from lib.ae_generator import Generator

if __name__=="__main__":
    args = sys.argv
    params = None
    try:
        root = str(args[1])
        num = int(args[2])
        size = int(args[3])
        params = (root, num, size)
    except:
        print("Wrong Input!")
        print("Example: python generate_vague_athletes.py ./examples/avatars_vague_athletes/ 10 96")
        print("where 10 represents 10 output avatars, and 96 represents the size of images")

    if params is not None:
        root, num, size = params
        generator = Generator("./models/ae_decoder_20211009_1.pth", 8)
        for i in trange(num):
            filename = os.path.join(root, str(i+1)+".png")
            generator.generate_png(filename, size)
