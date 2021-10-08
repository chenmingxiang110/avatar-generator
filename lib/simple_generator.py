import cv2
import numpy as np

def generate_githublike():
    color = np.random.random([1,1,3]) * 0.5
    face = np.random.random([5,5])
    T = sorted(face.reshape([-1]), reverse=True)[np.random.randint(7)+6]
    face[:,3] = face[:,1]
    face[:,4] = face[:,0]
    face = np.clip((1 - np.repeat(face>T, 3).reshape(5,5,3) * color) * 255, 0, 255)
    face = face.astype(np.uint8)
    return face
