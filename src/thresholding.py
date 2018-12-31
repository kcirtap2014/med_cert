import numpy as np
import cv2
from helper_functions import detect_peaks
import pdb

class Thresholding:
    def __init__(self, image, verbose=False, gauss_blur_kernel=(1,1),
                adapt_thresh_blocksize=19, adapt_thresh_C=20, option=0):
        self.image = np.array(image)
        self.gauss_blur_kernel = gauss_blur_kernel
        self.adapt_thresh_blocksize = adapt_thresh_blocksize
        self.adapt_thresh_C= adapt_thresh_C
        self.method = []
        self.ret = 0
        self.option = option

    def peak_detection(self, mpd = 10):
        n, _ = np.histogram(self.image.ravel(), bins=256)
        peakInd = detect_peaks(n, mpd = mpd)
        # take only the first two highest peaks to test for comparison
        # add in the last peak that is undetectable by this algo
        # print(list(peakInd)+[255])
        peaks = sorted(n[list(peakInd)+[255]], reverse=True)[:2]
        argmaxpeak = np.argmax(n[list(peakInd)+[255]])
        if len(peaks)==1:
            peaks.append(1.)

        return peaks, argmaxpeak

    def run(self, verbose=False):
        """
        loop through options or just apply one
        """

        peaks, argmaxpeak = self.peak_detection()

        if peaks[0]<100*peaks[1]:# or not argmaxpeak==255:
            ## QUESTION: or not peak_max==255:
            self.method.append('Adaptive_thresholding')
            self.method.append('Otsu_thresholding')
        else:
            # can be generated to more use cases
            self.method.append('Otsu_thresholding')
            self.method.append('Adaptive_thresholding')

        if self.method[self.option]=='Adaptive_thresholding':
            # peak_max==255 takes care of cases with heavy shadow
            # use adaptive when peaks are of the same order of magnitude

            if verbose:
                print(peaks, self.method[self.option])

            self.image = cv2.adaptiveThreshold(self.image,
                                            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            self.adapt_thresh_blocksize,
                                            self.adapt_thresh_C)

        elif self.method[self.option]=='Otsu_thresholding':

            if verbose:
                print(peaks, self.method[self.option])
            # use otsu when there is a distinctive amplitude (1 order of magnitude difference at least)
            #blur = cv2.GaussianBlur(self.image, self.gauss_blur_kernel, 0)
            self.ret, self.image = cv2.threshold(self.image, 0, 255,
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if verbose:
                print("Otsu threshold:%d" %self.ret)
