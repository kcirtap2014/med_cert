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
from copy import copy
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

    for i, file in enumerate(sorted_file_list):
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

        # Setting variables for the loop
        option = 0
        img2 = copy(img)
        score_total = pd.DataFrame(data=[[0,0,0,0]], 
                                   columns=df_exp.columns.tolist()[5:9],
                                   index = [i])
        
        # Attempting to read text from images with various processing options
        while((score_total.sum(axis=1).values != 4) and (option < 6)):
            img, img2 = thresholding(img, img2, DIR_PATH, option)
            im = trim(img2)
            txt_img = pytesseract.image_to_string(im)
            txt = text_preprocess(txt_img)

            if verbose:
                print("try ", option + 1,":", txt)

            temp = keyword_lookup(i, df_exp, filename, txt,
                                  BEGIN_DATE, C_KEYWORDS)
            score = temp.iloc[:,5:9]*1
            score_total += score

            if verbose:
                print(score_total)

            score_total.replace(2, 1, inplace=True)
            option += 1

        for f in score_total.columns:
            if(int(score_total[f][:1])==1):
                temp[f][:1]=True
            else:
                temp[f][:1]=False

        df_exp = df_exp.append(temp)

    if verbose:
        print(df_exp)
        # Removing temporary pdf file
    try:
        os.remove(DIR_PATH+"/temp.pdf")
    except:
        pass

    df_exp.to_csv(DIR_PATH + "/df_test.csv", index=False)
