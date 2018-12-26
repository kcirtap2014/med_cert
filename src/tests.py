from thresholding import Thresholding
from segmentation import Segmentation
import matplotlib.pylab as plt
import matplotlib.patches as patches
import os
from config import DIR_PATH, CERT_PATH, RETAINED_FILE_PATH, SEGMENTED_PATH, A4_100DPI
from pdf2image import convert_from_path
import pickle
import sys
from PIL import Image
import pdb
import cv2
import numpy as np
from helper_functions import wordSegmentation
from image_preprocessing import ImagePreprocessing

class Test:
    def __init__(self, image):
        self.image = image
        self.title = []
        self.images = []

    def t_thresholding(self):
        thresh = Thresholding(self.image)
        thresh.run()
        self.image = thresh.image
        self.title.append('Image thresholding: %s' %thresh.method)
        self.images.append(self.image)

    def t_segmentation(self):
        self.title.append('Before processing')
        self.images.append(self.image)
        im_proc = ImagePreprocessing(self.image, p_hough = True)
        im_proc.process()
        self.image = im_proc.image
        new_comp_ = wordSegmentation(self.image, kernelSize=25,
                                sigma=11, theta=7, minArea=0)

        #segmentation = Segmentation(self.image, p_hough=True)
        #segmentation.run()
        #self.image = segmentation.image
        self.title.append('Segmentation')
        self.images.append(self.image)

        #self.images.append(segmentation.image_b)
        #self.segmentation = segmentation
        #self.segmentation.new_components_ = new_comp_
        self.new_comp_ = new_comp_


    def _plot(self):
        n_row = len(self.images)
        n_col = 1
        self.fig, axes = plt.subplots(n_col, n_row, figsize=(18,12))

        for i, ax in enumerate(axes.flatten()):
            ax.imshow(self.images[i], cmap='gray')

            if self.title[i] == 'Segmentation':

                for dim, segment in self.new_comp_:
                    x, y, w, h = dim

                        #top left corner, and bottom right conner
                    rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                                 edgecolor='r',
                                                 facecolor='none')
                    ax.add_patch(rect)

            elif self.title[i] == 'Text lines':
                for x, y, w, h in self.segmentation.bboxes_arlsa:
                    #top left corner, and bottom right conner
                    rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                             edgecolor='r',
                                             facecolor='none')
                    ax.add_patch(rect)
            else:
                ax.imshow(self.images[i], cmap='gray')

        plt.tight_layout()


    def run(self):
        #self.t_thresholding()
        self.t_segmentation()
        self._plot()

if __name__ == '__main__':

    try:
        pickle_file = os.path.join(RETAINED_FILE_PATH, sys.argv[1])
    except IndexError:
        pickle_file = os.path.join(RETAINED_FILE_PATH, 'retained_file_score_0')

    with open(os.path.join(DIR_PATH, pickle_file),'rb') as fp:
        test_files = pickle.load(fp)

    for filename in test_files:
        print(filename)
        src = os.path.join(CERT_PATH, filename)

        if filename.endswith(".pdf"):
            img = convert_from_path(src, fmt="png", dpi=200)[0].convert('L')
        else:
            img = Image.open(src).convert('L')

        img = cv2.resize(np.asarray(img), A4_100DPI)
        test = Test(img)
        test.run()
        test.fig.savefig(os.path.join(SEGMENTED_PATH, filename.split('.')[0] +'.pdf'))
