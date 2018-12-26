import cv2
import numpy as np
from skimage.transform import resize


class Resize:

    def __init__(self, imgSize):
        self.imgSize = imgSize

    def transform(self, img):
        """
        Image resizeing given an input size
        """
        # create target image and copy sample image into it
        (wt, ht) = self.imgSize
        (h, w) = img.shape
        fx = w / wt
        fy = h / ht
        f = max(fx, fy)
        newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1)) # scale according to f (result at least 1 and at most wt or ht)
        #img = rescale(img, newSize)
        img = resize(img, (wt,ht), anti_aliasing=True)
        print(img.shape)
        #target = np.ones([ht, wt]) * 255
        #target[0:newSize[1], 0:newSize[0]] = img

        # transpose for TF
        #img = cv2.transpose(target)

        # normalize
        #(m, s) = cv2.meanStdDev(img)
        #m = m[0][0]
        #s = s[0][0]
        #img = img - m
        #img = img / s if s>0 else img

        return img
