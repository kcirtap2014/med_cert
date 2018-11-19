from urllib.request import urlretrieve
from bs4 import BeautifulSoup
import os
import re
import shutil
import pdb
import pandas as pd
import numpy as np
import pytesseract
import argparse
from PIL import Image
from pdf2image import convert_from_path
import warnings
import cv2 as cv
import PyPDF2
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
import time
from helper_functions import (rotation, text_preprocess, trim,
                              keyword_lookup, extract_name, find_match)
from config import dir_path, c_keywords, begin_date, chrome_driver_path
Image.MAX_IMAGE_PIXELS = 1000000000

warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", help="Activate verbose",type=bool, nargs='?',
                    const=True, default=False)
args = parser.parse_args()

temp_path = dir_path + "/temp_file/"
last_year_list_path = dir_path +"/list2018/"
output_path = dir_path +"/output/"

# check if fig_path exists
if not os.path.exists(temp_path):
    os.makedirs(temp_path)

if __name__ == "__main__":
    verbose  = bool(args.verbose)
    #set up dataframe
    url = 'https://www.topchrono.biz/organisateur/index.html'
    values = {'login': 'inscriptionstvalentin@frontrunnersparis.org',
              'pwd': '2017STVALMDP'}
    url2= 'https://www.topchrono.biz/espace-organisateur.php?login=Logstvalen2019!&pwd=Passtvalen2019!&redirect=espace-organisateur.php'

    # some flags
    timeout = 3
    l_duo = False

    # set up browser
    browser = webdriver.Chrome(chrome_driver_path)
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
    df_save = ["df_10km.csv", "df_5km_cupid.csv", "df_5km_duo.csv"]

    for i in range(2,3):
        # re-initialise df_prod each time
        from config import df_prod

        if i==2:
            #only activate when it's for duo category
            l_duo = True

        quote_page = 'https://www.topchrono.biz/IEL_orgaInscriptionCoureur.php?idCourse=1231%s&idInsc=740%s&changeTri=4' % (str(fchar[i]), str(schar[i]))

        list_last_year = last_year_list_path + "inscriptions_%d.xls" %(i+1)
        l_part = 0
        df_last_year = pd.read_excel(list_last_year)
        df_savename = '/df_%d' %(i+1)

        #&changeTri=4 if want to arange by date
        browser.get(quote_page)
        xpath = "//a[contains(text(), 'Cliquez ici pour afficher tout')]"
        nxt_btn = WebDriverWait(browser, timeout).until(EC.presence_of_element_located((By.XPATH, xpath))).click()
        #browser.find_elements_by_xpath(xpath)[0].click()
        html_source = browser.page_source
        soup = BeautifulSoup(html_source, 'html5lib')
        #img_list = soup.findAll(attrs={'title' : re.compile("^Afficher")})
        # look only for

        litige_img_list = soup.findAll(attrs={'class':'litige',
                                              'id':re.compile("^tr1")})

        #litige_patrick= soup.findAll(attrs={'id':re.compile("tr1_1479387")})

        #for j, element in enumerate(litige_patrick):#litige_img_list[-1:]):
        for j, element in enumerate(litige_img_list):

            #id_url = re.compile(r'\"\/IEL_orgaInscriptionCoureur.php\?(.*)\"').findall(str(element))[0]
            id = element.attrs.get('id').split('_')[-1]
            img_url = re.compile(r'\'(https[\s\S]*?)\'').findall(str(element))

            for k, i_url in enumerate(img_url):
                if l_duo:
                    l_part = k + 1

                filename = i_url.split("/")[-1]
                print("%d:%s"%(j,filename))

                credentials = filename#extract_name(filename)[1:]
                #valid_registration, error = find_match(credentials, df_last_year, l_part)

                #if error==1:
                #    temp = credentials
                #    credentials = [temp[1]] + [temp[0]] + temp[2:]

                if True:
                    src = temp_path + filename
                    i_url = i_url.replace(' ','%20')
                    urlretrieve(i_url, src)

                    try:
                        if not filename.endswith(".pdf"):
                            src_pdf = temp_path + filename.split('.')[0] + ".pdf"

                            if not os.path.isfile(src_pdf):
                                # convert other ext files to pdf if not found in dir_path_pdf
                                im_temp = Image.open(src).convert('L')
                                # crop white space
                                im_temp = trim(im_temp)
                                im_temp = rotation(im_temp)
                                im_temp.save(src_pdf, "pdf", optimize=True, quality=85)

                            src = src_pdf

                        PyPDF2.PdfFileReader(open(src, "rb"))
                        img = convert_from_path(src, fmt="png")[0].convert('L')
                        # crop white space
                        im = trim(img)
                        #mat_img = np.asarray(img)
                        # get rid of salt and pepper noise
                        #im = cv.medianBlur(mat_img, 3)

                        txt_img = pytesseract.image_to_string(im)

                        txt = text_preprocess(txt_img)
                        if verbose:
                            print("try 1:", txt)

                        temp = keyword_lookup(j, df_prod, credentials, txt, begin_date,
                                              c_keywords, l_prod=True)
                        score_total = temp.iloc[:,6:]*1
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

                            temp = keyword_lookup(j, df_prod, filename, txt,
                                                  begin_date, c_keywords, l_prod=True)
                            score=temp.iloc[:,6:]*1
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

                            temp = keyword_lookup(j, df_prod, filename, txt,
                                                  begin_date, c_keywords, l_prod=True)
                            score = temp.iloc[:,6:]*1
                            #print("Thumbnail")
                            score_total += score
                            score_total.replace(2, 1, inplace=True)

                        #Try 3 : reducing image quality by savind a thumbnail and reloading
                        if(score_total.sum(axis=1).values != 4):

                            img.thumbnail((1000,1000),Image.ANTIALIAS)
                            # Creating a temporary pdf save
                            img.save(temp_path+"/temp.pdf")

                            img = convert_from_path(temp_path+"/temp.pdf", fmt="png")[0].convert('L')
                            im = trim(img)
                            txt_img = pytesseract.image_to_string(im)
                            txt = text_preprocess(txt_img)
                            if verbose:
                                print("try 4:", txt)

                            temp = keyword_lookup(j, df_prod, filename, txt,
                                                  begin_date, c_keywords, l_prod=True)
                            score = temp.iloc[:,6:]*1
                            #print("Reduce")
                            score_total += score
                            score_total.replace(2, 1, inplace=True)

                        # Try 4 : adaptative thresholding on the reduced quality image
                        if(score_total.sum(axis=1).values!=4):

                            IMG0=np.array(img)
                            thresh=cv.adaptiveThreshold(IMG0,255,
                                                        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                        cv.THRESH_BINARY,15,11)
                            img_thresh = Image.fromarray(thresh)
                            im = trim(img_thresh)
                            txt_img = pytesseract.image_to_string(im)
                            txt = text_preprocess(txt_img)
                            if verbose:
                                print("try 5:", txt)

                            temp = keyword_lookup(j, df_prod, filename, txt, begin_date,
                                                  c_keywords, l_prod=True)
                            score = temp.iloc[:,6:]*1
                            #print("Reduced Thresholding : ")
                            score_total += score
                            score_total.replace(2,1,inplace=True)

                        #Try 5 : Making a thumbnail of the reduced image
                        if(score_total.sum(axis=1).values!=4):

                            img_thresh.thumbnail((1000,1000))
                            im = trim(img_thresh)
                            txt_img = pytesseract.image_to_string(im)
                            txt = text_preprocess(txt_img)
                            if verbose:
                                print("try 6:", txt)

                            temp = keyword_lookup(j, df_prod, filename, txt,
                                                  begin_date, c_keywords, l_prod=True)
                            score = temp.iloc[:,6:]*1
                            #print("Reduced thumbnail : ")
                            score_total += score
                            score_total.replace(2, 1, inplace=True)

                        # Removing temporary pdf file
                        for f in score_total.columns:
                            if(int(score_total[f][:1])==1):
                                temp[f][:1]=True
                            else:
                                temp[f][:1]=False


                        keep_columns = ["C_Nom","C_Prenom","C_Date","C_Mention"]
                        score = temp[keep_columns].sum(axis=1).values
                        print("Score:%d" %score)

                        if score!=4:
                            # keep only cases that are different from 4 for
                            # improvement purpose
                            df_prod = df_prod.append(temp)

                        if verbose:
                            print(df_prod)

                        if score == 4:
                            print("Certificat est validé")

                            if l_duo:
                                if l_part ==1:
                                    id_xpath = "chkLitige_%s" %str(id)
                                    xpath_valid = "//*[@id=\"ConfirmDelitige_%s\"]/div[2]" %str(id)
                                elif l_part==2:
                                    id_xpath = "chkLitige2_%s" %str(id)
                                    xpath_valid = "//*[@id=\"ConfirmDelitige2_%s\"]/div[2]" %str(id)
                            else:
                                id_xpath = "chkLitige_%s" %str(id)
                                xpath_valid = "//*[@id=\"ConfirmDelitige_%s\"]/div[2]" %str(id)
                            # first click to open the menu
                            id_button = browser.find_elements_by_id(id_xpath)[0]

                            if bool(id_button.get_attribute('checked')):
                                # click only when it's checked
                                ActionChains(browser).move_to_element(id_button).click(id_button).perform()
                                # look for the confirm delitige id
                                #try:
                                element = WebDriverWait(browser, timeout).until(EC.presence_of_element_located((By.XPATH, xpath_valid)))
                                # second click to ConfirmDelitige
                                ActionChains(browser).move_to_element(element).click(element).perform()

                            #except TimeoutException:
                            #    pass

                        else:
                            print("Certificat non validé, veuillez le valider manuellement")
                    except (PyPDF2.utils.PdfReadError, OSError, TypeError):
                        extracted = extract_name(credentials)
                        temp = pd.DataFrame(
                            [extracted + [credentials, False, False, False, False]],
                            columns=df_prod.columns,
                            index=[j])
                        df_prod = df_prod.append(temp)
                        print("invalid PDF file")
            #writer = pd.ExcelWriter(os.join.path(output_path, df_save[i]))
            #df_prod.to_excel(writer, index=False)
            df_prod.to_csv(os.path.join(output_path, df_save[i]), index=False)

    browser.quit()
    shutil.rmtree(temp_path)
