import cv2
import numpy as np
from thresholding import Thresholding
from helper_functions import (rotation, hough_line_transform)

class ImagePreprocessing:

    def __init__(self, image, verbose=False, p_hough = False, minLineLength = 40,
                 maxLineGap = 10, linewidth=1, graph_line=True, morphology=True,
                 option=0, morphology_kernel=(3,3)):
        self.verbose = verbose
        self.image = np.array(image)
        self.p_hough = p_hough
        self.minLineLength = minLineLength
        self.maxLineGap = maxLineGap
        self.linewidth = linewidth
        self.graph_line = graph_line
        self.morphology = morphology
        self.option = option
        self.morphology_kernel = morphology_kernel
        # count loops

    def process(self):

        self.image = rotation(self.image)

        if self.graph_line:
            # Step 1: Graphical line removal
            self.image = hough_line_transform(self.image, p_hough=self.p_hough,
                                   minLineLength=self.minLineLength,
                                   maxLineGap=self.maxLineGap,
                                   linewidth=self.linewidth)
        # thresholding
        thresh = Thresholding(self.image, option=self.option)
        thresh.run(verbose=self.verbose)
        self.image = thresh.image

        # morphology closing
        if self.morphology:
            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           self.morphology_kernel)

            self.image = cv2.morphologyEx(cv2.bitwise_not(self.image),
                                          cv2.MORPH_CLOSE, se, iterations=1)

            # convert back to black foreground
            self.image = cv2.bitwise_not(self.image)
