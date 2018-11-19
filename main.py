import pandas as pd
import numpy as np
import seaborn as sns
import pytesseract
import os
import argparse
from PIL import Image
from pdf2image import convert_from_path
from skimage.filters import threshold_otsu, threshold_mean #No longer used?
import warnings
import pickle #No longer used?
import cv2 as cv
import pdb
from helper_functions import thresholding
from config import DIR_PATH, df_exp, ext, file_list, C_KEYWORDS, BEGIN_DATE

#### To use directly in python , set working directory to contain helper_functions.py and the directory containing certificates
#### os.chdir("Desktop/Certificats/Cert_Recognition/")

### To use pytesseract after standard windows installation:
### pytesseract.pytesseract.tesseract_cmd='C:/Program Files (x86)/Tesseract-OCR/tesseract'

from helper_functions import rotation, text_preprocess, trim, keyword_lookup

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_style("ticks")
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", help="Activate verbose",type=bool, nargs='?',
                    const=True, default=False)
args = parser.parse_args()

if __name__ == '__main__':
    verbose  = bool(args.verbose)

    # some flags
    dir_pdf_path = DIR_PATH + "/pdf/"

    # load certificates
    cert_dir = DIR_PATH + '/TestCertificats/'

    #keywords_preprocessed = []
    # sort by alphabetical order
    sorted_file_list = sorted(file_list)

    for i, file in enumerate(sorted_file_list[:2]):
        filename = os.fsdecode(file)
        src = os.path.join(str(cert_dir), filename)
        print("%d:%s"%(i,filename))

        if not filename.endswith(".pdf"):
            src_pdf = dir_pdf_path + filename.split('.')[0] + ".pdf"

            if not os.path.isfile(src_pdf):
                # convert other ext files to pdf if not found in DIR_PATH_pdf
                im_temp = Image.open(src).convert('L')
                # crop white space
                #im_temp = trim(im_temp)
                im_temp = rotation(im_temp)
                im_temp.save(src_pdf, "pdf", optimize=True, quality=85)

            img = convert_from_path(src_pdf, fmt="png")[0].convert('L')
        else:
            img = convert_from_path(src, fmt="png")[0].convert('L')

        # crop white space
        im = trim(img)
        mat_img = np.asarray(im)
        # get rid of salt and pepper noise
        im = cv.medianBlur(mat_img, 3)

        # Reading text, searching for keywords
        txt_img = pytesseract.image_to_string(im)
        txt = text_preprocess(txt_img)
        temp = keyword_lookup(i, df_exp, filename, txt, BEGIN_DATE, C_KEYWORDS)
        if verbose:
            print("try 1: ", txt)
        # Keeping track of validated keywords in numeric format
        score_total = temp.iloc[:,5:9]*1
        #temp.iloc[:,5:9].sum(axis=1).values
        # If keywords are missing, applying transformations to try and find them

        # Try 1 : Adaptative thresholding
        if (score_total.sum(axis=1).values != 4):

            IMG0 = np.array(img)
            thresh = cv.adaptiveThreshold(IMG0,255,
                                        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv.THRESH_BINARY,15,11)
            img_thresh = Image.fromarray(thresh)
            im = trim(img_thresh)

            # pytesseract will sometimes crash if the image is too big (but only after thresholding for some reason)
            if (np.array(img).shape[0]*np.array(img).shape[1] > 80000000):
                im.thumbnail((2000,2000),Image.ANTIALIAS)

            txt_img=pytesseract.image_to_string(im)
            txt = text_preprocess(txt_img)
            if verbose:
                print("try 2:", txt)

            temp = keyword_lookup(i, df_exp, filename, txt,
                                  BEGIN_DATE, C_KEYWORDS)
            score=temp.iloc[:,5:9]*1
            #print("Thresholding")
            #Adding newly validated mentions to the tracker
            score_total += score
            score_total.replace(2, 1, inplace=True)

        #Try 2 : Reducing image to a thumbnail
        if (score_total.sum(axis=1).values!=4):

            img_thresh.thumbnail((1000,1000))
            im = trim(img_thresh)
            txt_img = pytesseract.image_to_string(im)
            txt = text_preprocess(txt_img)
            if verbose:
                print("try 3:", txt)
            temp = keyword_lookup(i, df_exp, filename, txt,
                                  BEGIN_DATE, C_KEYWORDS)
            score = temp.iloc[:,5:9]*1
            #print("Thumbnail")
            score_total += score
            score_total.replace(2, 1, inplace=True)

        #Try 3 : reducing image quality by savind a thumbnail and reloading
        if(score_total.sum(axis=1).values != 4):

            img.thumbnail((1000,1000),Image.ANTIALIAS)
            # Creating a temporary pdf save
            img.save(DIR_PATH+"/temp.pdf")
            img = convert_from_path(DIR_PATH+"/temp.pdf", fmt="png")[0].convert('L')
            im = trim(img)
            txt_img = pytesseract.image_to_string(im)
            txt = text_preprocess(txt_img)
            if verbose:
                print("try 4:", txt)
            temp = keyword_lookup(i, df_exp, filename, txt,
                                  BEGIN_DATE, C_KEYWORDS)
            score = temp.iloc[:,5:9]*1
            #print("Reduce")
            score_total += score
            score_total.replace(2, 1, inplace=True)

        # Try 4 : adaptative thresholding on the reduced quality image
        if(score_total.sum(axis=1).values!=4):

            IMG0=np.array(img)
            thresh=cv.adaptiveThreshold(IMG0,255,
                                        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv.THRESH_BINARY,15,11)
            img_thresh=Image.fromarray(thresh)
            im = trim(img_thresh)
            txt_img=pytesseract.image_to_string(im)
            txt = text_preprocess(txt_img)
            if verbose:
                print("try 5:", txt)
            temp = keyword_lookup(i, df_exp, filename, txt, BEGIN_DATE, C_KEYWORDS)
            score=temp.iloc[:,5:9]*1
            #print("Reduced Thresholding : ")
            score_total+=score
            score_total.replace(2,1,inplace=True)

        #Try 5 : Making a thumbnail of the reduced image
        if(score_total.sum(axis=1).values!=4):

            img_thresh.thumbnail((1000,1000))
            im = trim(img_thresh)
            txt_img = pytesseract.image_to_string(im)
            txt = text_preprocess(txt_img)
            if verbose:
                print("try 6:", txt)
            temp = keyword_lookup(i, df_exp, filename, txt,
                                  BEGIN_DATE, C_KEYWORDS)
            score = temp.iloc[:,5:9]*1
            #print("Reduced thumbnail : ")
            score_total += score
            score_total.replace(2, 1, inplace=True)

        # Removing temporary pdf file
        for f in score_total.columns:
            if(int(score_total[f][:1])==1):
                temp[f][:1]=True
            else:
                temp[f][:1]=False

        df_exp = df_exp.append(temp)

    if verbose:
        print(df_exp)

    try:
        os.remove(DIR_PATH+"/temp.pdf")
    except:
        pass

    df_exp.to_csv(DIR_PATH + "/df_test.csv", index=False)
