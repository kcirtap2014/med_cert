from urllib.request import urlretrieve
from bs4 import BeautifulSoup
import os
import re
import shutil
import pdb
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from skimage.filters import threshold_otsu, threshold_mean
import warnings
import PyPDF2
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
from helper_functions import rotation, text_preprocess, trim, keyword_lookup

warnings.filterwarnings("ignore", category=UserWarning)
dir_path = os.getcwd()
temp_path = dir_path + "/temp_file/"

# check if fig_path exists
if not os.path.exists(temp_path):
    os.makedirs(temp_path)

if __name__ == "__main__":

    #set up dataframe
    df = pd.DataFrame(columns=[
        "R_Date", "Nom", "Prenom", "BirthYear", "Ext",
        "FileName", "C_Nom","C_Prenom", "C_Date", "C_Mention"
    ])

    url = 'https://www.topchrono.biz/organisateur/index.html'
    values = {'login': 'inscriptionstvalentin@frontrunnersparis.org',
              'pwd': '2017STVALMDP'}
    url2= 'https://www.topchrono.biz/espace-organisateur.php?login=Logstvalen2019!&pwd=Passtvalen2019!&redirect=espace-organisateur.php'

    timeout = 3

    # set up browser
    browser = webdriver.Safari()
    browser.get(url)
    # login
    browser.find_element_by_name('login').send_keys(values['login'])
    browser.find_element_by_name('pwd').send_keys(values['pwd'])
    inscr_btn = WebDriverWait(browser, timeout).until(EC.presence_of_element_located((By.NAME, 'connecter')))
    inscr_btn.click()
    #browser.find_element_by_name('connecter').click()
    browser.get(url2)
    #_ = session_requests.get(url2, headers = dict(referer = url2))
    fchar = np.arange(7,10)
    schar = np.arange(3,6)


    for i in range(3):
        quote_page = 'https://www.topchrono.biz/IEL_orgaInscriptionCoureur.php?idCourse=1231%s&idInsc=740%s&changeTri=4' % (str(fchar[i]), str(schar[i]))
        #&changeTri=4 if want to arange by date
        browser.get(quote_page)
        xpath = "//a[contains(text(), 'Cliquez ici pour afficher tout')]"
        nxt_btn = WebDriverWait(browser, timeout).until(EC.presence_of_element_located((By.XPATH, xpath)))
        nxt_btn.click()
        time.sleep(timeout)
        #browser.find_elements_by_xpath(xpath)[0].click()
        html_source = browser.page_source
        soup = BeautifulSoup(html_source, 'html5lib')
        #img_list = soup.findAll(attrs={'title' : re.compile("^Afficher")})
        # look only for
        litige_img_list = soup.findAll(attrs={'class':'litige',
                                              'id':re.compile("^tr1")})

        litige_patrick= soup.findAll(attrs={'id':re.compile("tr1_1479387")})

        for j, element in enumerate(litige_patrick):#litige_img_list[-1:]):

            id_url = re.compile(r'\"\/IEL_orgaInscriptionCoureur.php\?(.*)\"').findall(str(element))[0]
            id = id_url.split("=")[-1]
            img_url = re.compile(r'\'([\s\S]*?)\'').findall(str(element))[0]
            filename = img_url.split("/")[-1]

            src = temp_path + filename
            urlretrieve(img_url, src)
            print("%d:%s"%(j,filename))

            try:
                if not filename.endswith(".pdf"):
                    src_pdf = temp_path + filename.split('.')[0] + ".pdf"

                    if not os.path.isfile(src_pdf):
                        # convert other ext files to pdf if not found in dir_path_pdf
                        im_temp = Image.open(src).convert('L')
                        # crop white space
                        #im_temp = trim(im_temp)
                        im_temp = rotation(im_temp)
                        im_temp.save(src_pdf, "pdf", optimize=True, quality=85)

                    src = src_pdf

                PyPDF2.PdfFileReader(open(src, "rb"))
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

            except (PyPDF2.utils.PdfReadError, OSError):
                txt = ""
                print("invalid PDF file")

            df = keyword_lookup(j, df, filename, txt, '17/2/2018',
                                        l_prod=True)
            #os.remove(temp_path+filename)
            keep_columns = ["C_Nom","C_Prenom","C_Date","C_Mention"]
            df["Score"] = df[keep_columns].sum(axis=1).values

            if df["Score"].values == 4:
                print("Certificat est validé")
                valid_path = ".//input[contains(@id, \"chkLitige_%s\")]" %str(id)
                browser.find_elements_by_id("chkLitige_%s" %str(id))[0].click()
                #browser.find_elements_by_xpath(valid_path)[0].click()
                # all correct, then cancel delitige
    browser.quit()
    shutil.rmtree(temp_path)
    df.to_csv(dir_path + "/df_real.csv", index=False)
