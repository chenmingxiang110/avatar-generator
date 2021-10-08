import os
import cv2
import sys
import numpy as np

from lib.simple_generator import generate_githublike

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
        print("Example: python generate_githublike.py ./examples/avatars_githublike/ 10 96")
        print("where 10 represents 10 output avatars, and 96 represents the size of images")
    
    if params is not None:
        root, num, size = params
        for i in range(num):
            filename = os.path.join(root, str(i+1)+".png")
            inner_size = int(size * 5 / 6)
            pad = int(size / 12)
            face = cv2.resize(generate_githublike(), (inner_size, inner_size), interpolation=cv2.INTER_NEAREST)
            img = (np.ones([size,size,3])*255).astype(np.uint8)
            img[pad:pad+inner_size, pad:pad+inner_size] = face
            cv2.imwrite(filename, img)