import pandas as pd
import numpy as np
import seaborn as sns
import pytesseract
import os
from PIL import Image
from pdf2image import convert_from_path
from skimage.filters import threshold_otsu, threshold_mean #No longer used?
import warnings
import pickle #No longer used?
import cv2 as cv
import pdb
from config import dir_path, df_exp, ext, file_list, c_keywords, begin_date

#### To use directly in python , set working directory to contain helper_functions.py and the directory containing certificates
#### os.chdir("Desktop/Certificats/Cert_Recognition/")

### To use pytesseract after standard windows installation:
### pytesseract.pytesseract.tesseract_cmd='C:/Program Files (x86)/Tesseract-OCR/tesseract'

from helper_functions import rotation, text_preprocess, trim, keyword_lookup

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_style("ticks")

if __name__ == '__main__':

    # some flags
    dir_pdf_path = dir_path + "/pdf/"

    # load certificates
    cert_dir = dir_path + '/TestCertificats/'

    # prepare an empty DataFrame
    #df_exp = pd.DataFrame(columns=[
    #    "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom",
    #    "C_Prenom", "C_Date", "C_Mention"])

    # define file_list
    #ext = tuple(['pdf', 'jpg', 'jpeg', 'png'])
    #file_list = [f for f in os.listdir(cert_dir) if f.endswith(ext)]
    #c_keywords = [("athle", "running", "course", "pied", "compétition",
    #               "athlétisme", "semi-marathon", "marathon")]
    begin_date = '17/2/2018'
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
                # convert other ext files to pdf if not found in dir_path_pdf
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

        # Reading text, searching for keywords
        txt_img = pytesseract.image_to_string(im)
        txt = text_preprocess(txt_img)
        temp = keyword_lookup(i, df_exp, filename, txt, begin_date, c_keywords)

        # Keeping track of validated keywords in numeric format
        score_total = temp.iloc[:,5:9]*1
        #temp.iloc[:,5:9].sum(axis=1).values
        # If keywords are missing, applying transformations to try and find them

        # Try 1 : Adaptative thresholding
        if (score_total.transpose().iloc[:,0].sum() != 4):
            
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
            temp = keyword_lookup(i, df_exp, filename, txt,
                                  begin_date, c_keywords)
            score=temp.iloc[:,5:9]*1
            #print("Thresholding")
            #Adding newly validated mentions to the tracker
            score_total += score
            score_total.replace(2, 1, inplace=True)

        #Try 2 : Reducing image to a thumbnail
        if (score_total.transpose().iloc[:,0].sum()!=4):

            img_thresh.thumbnail((1000,1000))
            im = trim(img_thresh)
            txt_img = pytesseract.image_to_string(im)
            txt = text_preprocess(txt_img)
            temp = keyword_lookup(i, df_exp, filename, txt,
                                  begin_date, c_keywords)
            score = temp.iloc[:,5:9]*1
            #print("Thumbnail")
            score_total += score
            score_total.replace(2, 1, inplace=True)

        #Try 3 : reducing image quality by savind a thumbnail and reloading
        if(score_total.transpose().iloc[:,0].sum() != 4):

            img.thumbnail((1000,1000),Image.ANTIALIAS)
            # Creating a temporary pdf save
            img.save(dir_path+"/temp.pdf")
            img = convert_from_path(dir_path+"/temp.pdf", fmt="png")[0].convert('L')
            im = trim(img)
            txt_img = pytesseract.image_to_string(im)
            txt = text_preprocess(txt_img)
            temp = keyword_lookup(i, df_exp, filename, txt,
                                  begin_date, c_keywords)
            score = temp.iloc[:,5:9]*1
            #print("Reduce")
            score_total += score
            score_total.replace(2, 1, inplace=True)

        # Try 4 : adaptative thresholding on the reduced quality image
        if(score_total.transpose().iloc[:,0].sum()!=4):

            IMG0=np.array(img)
            thresh=cv.adaptiveThreshold(IMG0,255,
                                        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv.THRESH_BINARY,15,11)
            img_thresh=Image.fromarray(thresh)
            im = trim(img_thresh)
            txt_img=pytesseract.image_to_string(im)
            txt = text_preprocess(txt_img)
            temp = keyword_lookup(i, df_exp, filename, txt, begin_date, c_keywords)
            score=temp.iloc[:,5:9]*1
            #print("Reduced Thresholding : ")
            score_total+=score
            score_total.replace(2,1,inplace=True)

        #Try 5 : Making a thumbnail of the reduced image
        if(score_total.transpose().iloc[:,0].sum()!=4):

            img_thresh.thumbnail((1000,1000))
            im = trim(img_thresh)
            txt_img = pytesseract.image_to_string(im)
            txt = text_preprocess(txt_img)
            temp = keyword_lookup(i, df_exp, filename, txt,
                                  begin_date, c_keywords)
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

    print(df_exp)

    try:
        os.remove(dir_path+"/temp.pdf")
    except:
        pass

    df_exp.to_csv(dir_path + "/df.csv", index=False)


""" Cas à résoudre:

OK! Cas n°1,2: Date non détectée (probablement la date sous forme jj-mm-aaaa)

df_temp = pd.DataFrame(columns=[ "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom", "C_Prenom", "C_Date", "C_Mention"])
filename="nom_prenom_110718.pdf"
i=0
txt="courses a pied  a paris     cachet  le mer. 11-07-2018 :"
keyword_lookup(i, df_temp, filename, txt, '17/2/2018')


OK! df_temp = pd.DataFrame(columns=[ "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom", "C_Prenom", "C_Date", "C_Mention"])
filename="nom_prenom_050918.pdf"
i=0
txt="au samedi  paris, le 05-09-2018 j"
keyword_lookup(i, df_temp, filename, txt, '17/2/2018')


Cas n°3-7: Date non détectée (probablement date sours forme jj/mm/aa au lieu de jj/mm/aaaa)

OK! df_temp = pd.DataFrame(columns=[ "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom", "C_Prenom", "C_Date", "C_Mention"])
filename="nom_prenom_190618.pdf"
i=0
txt="samedi de 9h a 12h30 (en alternance)  paris, le 19/06/18."
keyword_lookup(i, df_temp, filename, txt, '17/2/2018')

OK! df_temp = pd.DataFrame(columns=[ "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom", "C_Prenom", "C_Date", "C_Mention"])
filename="nom_prenom_050918.pdf"
i=0
txt="competition  paris le 05/09/18"
keyword_lookup(i, df_temp, filename, txt, '17/2/2018')

OK! df_temp = pd.DataFrame(columns=[ "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom", "C_Prenom", "C_Date", "C_Mention"])
filename="nom_prenom_070918.pdf"
i=0
txt="en competiton natation en competiton  paris le : 07/09/18     membre d'un"
keyword_lookup(i, df_temp, filename, txt, '17/2/2018')

OK! df_temp = pd.DataFrame(columns=[ "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom", "C_Prenom", "C_Date", "C_Mention"])
filename="nom_prenom_220818.pdf"
i=0
txt="course a pied  en competition  paris le :22/08/18"
keyword_lookup(i, df_temp, filename, txt, '17/2/2018')

OK! df_temp = pd.DataFrame(columns=[ "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom", "C_Prenom", "C_Date", "C_Mention"])
filename="nom_prenom_060918.pdf"
i=0
txt="103081 8  le 06/09/18 certif"
keyword_lookup(i, df_temp, filename, txt, '17/2/2018')

OK! Cas n°8: Mention non détectée: marathon ne fait pas partie du dictionnaire des mentions

df_temp = pd.DataFrame(columns=[ "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom", "C_Prenom", "C_Date", "C_Mention"])
filename="nom_prenom_010101.pdf"
i=0
txt="pratique du sport en salle, en exterieur, a i'entrainement et a la competition du marathon "
keyword_lookup(i, df_temp, filename, txt, '17/2/2018')

Cas n°9: Date non détectée : la lettre û est mal lue -> autoriser un peu de flexibilité sur ce mot?

OK! df_temp = pd.DataFrame(columns=[ "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom", "C_Prenom", "C_Date", "C_Mention"])
filename="nom_prenom_280818.pdf"
i=0
txt="aparis 11 le 28 aodt 2018  je soussigne, dr"
keyword_lookup(i, df_temp, filename, txt, '17/2/2018')

txt="a paris ii le 28 aoiit 2018  je soussigne"

Cas n° 10 : Mention non détectée: faute de frappe

df_temp = pd.DataFrame(columns=[ "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom", "C_Prenom", "C_Date", "C_Mention"])
filename="nom_prenom_280818.pdf"
i=0
txt = "course apieds en competition"
keyword_lookup(i, df_temp, filename, txt, '17/2/2018')

"""
