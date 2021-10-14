import os
import sys
import cv2
import time
import torch
import numpy as np

from tqdm import trange, tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from lib.gan_generator import Generator

seed_path = "./models/avatar_04_cartoon_face_02.npy"
model_path = "./models/avatar_04_cartoon_face_02_decoder.pth"
generator = Generator(model_path, seed_path)

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
        print("Example: python generate_anime.py ./examples/avatars_anime/ 10 96")
        print("where 10 represents 10 output avatars, and 96 represents the size of images")
    
    if params is not None:
        root, num, size = params
        for i in trange(num):
            filename = os.path.join(root, str(i+1)+".png")
            img = cv2.resize(np.clip(generator.generate() * 255, 0, 255), (size, size)).astype(np.uint8)
            cv2.imwrite(filename, img)
            