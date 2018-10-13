import shutil
import pickle
import os
import numpy as np
import pytesseract
from PIL import Image
import cv2
import matplotlib.pylab as plt
from pdf2image import convert_from_path
from skimage.filters import threshold_otsu
from helper_functions import local_thresholding, wordSegmentation, trim
import matplotlib.patches as patches
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--plot", help="Activate plot",type=bool, nargs='?',
                    const=True, default=False)
args = parser.parse_args()

# run by typing python3 segmentation.py
if __name__ == '__main__':
    dir_path = os.getcwd()
    cert_dir = dir_path + '/TestCertificats/'

    # change the file name as you wish
    with open(dir_path +'/retained_file_score_3', 'rb') as fp:
        file_list_crnn = pickle.load(fp)

    for i, file in enumerate(file_list_crnn[2:3]):
        filename = os.fsdecode(file)
        src = os.path.join(str(cert_dir), filename)
        print("%d:%s"%(i,filename))

        if filename.endswith(".pdf"):
            # convert it to gray scale
            img = convert_from_path(src, fmt="png", dpi=200)[0].convert('L')

        else:
            # convert it to gray scale
            img = Image.open(src).convert('L')
        cropped_img = trim(img)
        mat_img = np.asarray(cropped_img)

        if False:
            # @Jérôme, thresholding et filtrages

            global_thresh = threshold_otsu(mat_img)
            mat_img_local, img_local = local_thresholding(mat_img, block_size=35,
                                                        offset=50)
            im = img.point(lambda p: np.logical_and(p > global_thresh, 255))
            mat_img_global = np.asarray(im)

        bb_tuple = wordSegmentation(mat_img, kernelSize=55, sigma=211,
                                    minArea=500)

    # remove folder where images are stored
    img_data_path = dir_path + "/img_data/"
    shutil.rmtree(img_data_path)
    os.makedirs(img_data_path)
    rects = []

    for i, tup in enumerate(bb_tuple):
        x,y,w,h = tup[0]
        rect = patches.Rectangle((x,y),w,h,linewidth=1, edgecolor='r',
                                 facecolor='none')
        rects.append(rect)
        img = Image.fromarray(tup[1])


        # @Jérôme: adjustment of the contrast
        pxmin = np.min(img)
        pxmax = np.max(img)
        imgContrast = (img - pxmin) / (pxmax - pxmin) * 255

        # increase line width
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(imgContrast, kernel, iterations = 1)

        txt = pytesseract.image_to_string(img,  lang='fra')
        cv2.imwrite(img_data_path + "/img%d.png"%i, img)

    if bool(args.plot):
        fig, ax = plt.subplots(figsize=(6,10))
        ax.imshow(mat_img, cmap='gray')

        for rect in rects:
            ax.add_patch(rect)
        plt.show()
