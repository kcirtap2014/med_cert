import shutil
import pickle
import os
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pylab as plt
from pdf2image import convert_from_path
from skimage.filters import threshold_otsu
from helper_functions import (local_thresholding, wordSegmentation,
                              trim, thresholding, rotation)
import matplotlib.patches as patches
import argparse
import cv2
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--plot", help="Activate plot",type=bool, nargs='?',
                    const=True, default=False)
parser.add_argument("--obs", help="obs data",type=bool, nargs='?',
                    const=True, default=False)
args = parser.parse_args()

# run by typing python3 segmentation.py
if __name__ == '__main__':


    if bool(args.obs):
        dir_path = '/Users/pmlee/Documents/FRP/Cert_Recognition/nouvel-obs/'
        input_path = dir_path + 'input/'
        img_path = dir_path + "words/"

        if os.path.exists(img_path):
            shutil.rmtree(img_path)

        elif not os.path.exists(img_path):
            # check if fig_path exists
            os.makedirs(img_path)

        file_list_crnn = os.listdir(input_path)

    else:
        dir_path = os.getcwd()
        dir_pdf_path = dir_path + "/pdf/"
        cert_dir = dir_path + '/TestCertificats/'
        input_path = dir_path + 'input/'
        img_path = dir_path + '/img_data/'

        with open(dir_path +'/retained_file_score_1', 'rb') as fp:
            file_list_crnn = pickle.load(fp)

    sorted_file_list_crnn = sorted(file_list_crnn)

    counter = 0
    for i, file in enumerate(sorted_file_list_crnn):
        filename = os.fsdecode(file)
        if bool(args.obs):
            src = os.path.join(input_path, filename)
        else:
            src = os.path.join(cert_dir, filename)
        print("%d:%s"%(i,filename))

        if filename.endswith(".pdf"):
            # convert it to gray scale
            img = convert_from_path(src, fmt="png", dpi=200)[0].convert('L')

        else:
            #src_pdf = dir_pdf_path + filename.split('.')[0] + ".pdf"
            # convert it to gray scale
            im_temp = Image.open(src).convert('L')

            img = rotation(im_temp)
            #im_temp.save(src_pdf, "pdf", optimize=True, quality=85)
            #img = convert_from_path(src_pdf, fmt="png")[0].convert('L')

        cropped_img = trim(img)
        #invert = cv2.bitwise_not(cropped_img)
        mat_img = np.asarray(cropped_img)


        #invert = cv2.bitwise_not(mat_img)
        #mat_img = thresholding(invert, option=0)
        bb_tuple = wordSegmentation(mat_img)#, kernelSize=55, sigma=211, minArea=500)

        # remove folder where images are stored
        #shutil.rmtree(img_path)
        #os.makedirs(img_path)
        rects = []

        if bool(args.obs):
            lot_path  = img_path + 'lot%d/' %i
            os.makedirs(lot_path)

        for j, tup in enumerate(bb_tuple):
            x,y,w,h = tup[0]
            rect = patches.Rectangle((x,y),w,h,linewidth=1, edgecolor='r',
                                     facecolor='none')
            rects.append(rect)
            #img = tup[1]
            img = Image.fromarray(tup[1])

            # @Jérôme: adjustment of the contrast
            #pxmin = np.min(img)
            #pxmax = np.max(img)
            #imgContrast = (img - pxmin) / (pxmax - pxmin) * 255

            # increase line width
            #kernel = np.ones((3, 3), np.uint8)
            #img = cv2.erode(img, kernel, iterations = 1)
            img = np.array(img)

            #txt = pytesseract.image_to_string(img)
            if bool(args.obs):
                cv2.imwrite(lot_path + "img%d.png" %(j), img)
            else:
                cv2.imwrite(img_path + "img%d.png" %(counter), img)
            counter+=1

        if bool(args.plot):
            fig, ax = plt.subplots(figsize=(6,10))
            ax.imshow(mat_img, cmap='gray')

            for rect in rects:
                ax.add_patch(rect)
            plt.show()
