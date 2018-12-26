from thresholding import Thresholding
from segmentation import Segmentation
import matplotlib.pylab as plt
import matplotlib.patches as patches
import os
from config import DIR_PATH, CERT_PATH, RETAINED_FILE_PATH, SEGMENTED_PATH
from pdf2image import convert_from_path
import pickle
import sys
from PIL import Image
import pdb

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
        segmentation = Segmentation(self.image, p_hough=False)
        segmentation.run()
        self.image = segmentation.image
        self.title.append('Segmentation')
        self.images.append(self.image)
        self.title.append('Text lines')
        self.images.append(segmentation.image_b)
        self.segmentation = segmentation

    def _plot(self):
        n_row = len(self.images)
        n_col = 1
        self.fig, axes = plt.subplots(n_col, n_row, figsize=(18,12))

        for i, ax in enumerate(axes.flatten()):
            ax.imshow(self.images[i], cmap='gray')

            if self.title[i] == 'Segmentation':
                for x, y, w, h in self.segmentation.new_components_:
                    #top left corner, and bottom right conner
                    rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                             edgecolor='r',
                                             facecolor='none')
                    ax.add_patch(rect)

            if self.title[i] == 'Text lines':
                for x, y, w, h in self.segmentation.bboxes_arlsa:
                    #top left corner, and bottom right conner
                    rect = patches.Rectangle((x, y), w, h, linewidth=1,
                                             edgecolor='r',
                                             facecolor='none')
                    ax.add_patch(rect)

        plt.tight_layout()


    def run(self):
        #self.t_thresholding()
        self.t_segmentation()
        self._plot()

if __name__ == '__main__':

    try:
        pickle_file = os.path.join(RETAINED_FILE_PATH, sys.argv[1])
    except IndexError:
        pickle_file = os.path.join(RETAINED_FILE_PATH, 'retained_file_score_1')

    with open(os.path.join(DIR_PATH, pickle_file),'rb') as fp:
        test_files = pickle.load(fp)

    for filename in test_files:
        print(filename)
        src = os.path.join(CERT_PATH, filename)

        if filename.endswith(".pdf"):
            img = convert_from_path(src, fmt="png", dpi=200)[0].convert('L')
        else:
            img = Image.open(src).convert('L')

        test = Test(img)
        test.run()
        test.fig.savefig(os.path.join(SEGMENTED_PATH, filename.split('.')[0] +'.pdf'))
