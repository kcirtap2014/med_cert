import pandas as pd
import numpy as np
import seaborn as sns
import pytesseract
import os
from PIL import Image
from pdf2image import convert_from_path
from skimage.filters import threshold_otsu, threshold_mean
import warnings
from helper_functions import rotation, text_preprocess, trim, keyword_lookup
import pickle

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_style("ticks")

if __name__ == '__main__':

    # some flags
    dir_path = os.getcwd()
    dir_pdf_path = dir_path +"/pdf/"

    # thresholding for segmenting objects form a background
    threshold = 200

    # load certificates
    cert_dir = dir_path + '/TestCertificats/'

    # prepare an empty DataFrame
    df_exp = pd.DataFrame(columns=[
        "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom",
        "C_Prenom", "C_Date", "C_Mention"
    ])

    # define file_list
    ext = tuple(['pdf', 'jpg', 'jpeg', 'png'])
    file_list = [f for f in os.listdir(cert_dir) if f.endswith(ext)]
    #file_list = ["Denizeaux_Paul_030918.pdf"]
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

        # performing thresholding using otsu thresholding which calculates the
        # optimum thershold pixel to be filtered
        mat_img = np.array(img.getdata()).reshape(img.size)
        txt_img = pytesseract.image_to_string(im,  lang='fra')
        txt = text_preprocess(txt_img)

        if False:
            # thresholding tests (@Jerôme, tu peux faire tes tests de thresholding
            # ou filtrages ici)
            thresh = threshold_otsu(mat_img)
            im_otsu = img.point(lambda p: p > thresh and 255)
            txt_img_otsu = pytesseract.image_to_string(im_otsu,  lang='fra')
            txt_otsu = text_preprocess(txt_img_otsu)

            thresh_mean = threshold_mean(mat_img)
            im_mean = img.point(lambda p: p > thresh_mean and 255)
            txt_img_mean = pytesseract.image_to_string(im_mean,  lang='fra')
            txt_mean = text_preprocess(txt_img_mean)
            print('\nResulting text (otsu):')
            print(txt_otsu)
            print("*"*30)
            print('\nResulting text (mean):')
            print(txt_mean)
            print("*"*30)
            print('\nResulting text:')
            print(txt)
        df_exp = keyword_lookup(i, df_exp, filename, txt, '17/2/2018')

    df_exp.to_csv(dir_path + "/df.csv", index=False)
