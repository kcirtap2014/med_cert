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
                              keyword_lookup, extract_name, find_match,
                              thresholding)
from config import DIR_PATH, C_KEYWORDS, BEGIN_DATE, CHROME_DRIVER_PATH
from config import TEMP_PATH, OUTPUT_PATH, keep_columns
from file_handling import FileHandling

Image.MAX_IMAGE_PIXELS = 1000000000

warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", help="Activate verbose",type=bool, nargs='?',
                    const=True, default=False)
args = parser.parse_args()

# check if fig_path exists
if not os.path.exists(TEMP_PATH):
    os.makedirs(TEMP_PATH)

if __name__ == "__main__":
    verbose  = bool(args.verbose)
    #set up dataframe
    url = 'https://www.topchrono.biz/organisateur/index.html'
    values = {'login': 'inscriptionstvalentin@frontrunnersparis.org',
              'pwd': 'Fr0ntRunStVal20ans!'}
    url2= 'https://www.topchrono.biz/espace-organisateur.php?login=Logstvalen2019!&pwd=Passtvalen2019!&redirect=espace-organisateur.php'

    # some flags
    timeout = 3
    l_duo = False

    # set up browser
    browser = webdriver.Chrome(CHROME_DRIVER_PATH)
    browser.get(url)
    # login
    browser.find_element_by_name('login').send_keys(values['login'])
    browser.find_element_by_name('pwd').send_keys(values['pwd'])
    inscr_btn = WebDriverWait(browser, timeout).until(EC.presence_of_element_located((By.NAME, 'connecter')))
    inscr_btn.click()
    browser.get(url2)
    fchar = np.arange(7,10)
    schar = np.arange(3,6)
    df_save = ["df_10km.csv", "df_5km_cupid.csv", "df_5km_duo.csv"]

    for i in range(3):
        # re-initialise fh.df each time
        SAVE_PATH = os.path.join(OUTPUT_PATH, df_save[i])
        fh = FileHandling(df_path = SAVE_PATH)
        fh.load_df()

        if i==2:
            #only activate when it's for duo category
            l_duo = True

        quote_page = 'https://www.topchrono.biz/IEL_orgaInscriptionCoureur.php?idCourse=1231%s&idInsc=740%s&changeTri=4' % (str(fchar[i]), str(schar[i]))

        l_part = 0
        browser.get(quote_page)
        xpath = "//a[contains(text(), 'Cliquez ici pour afficher tout')]"
        nxt_btn = WebDriverWait(browser, timeout).until(EC.presence_of_element_located((By.XPATH, xpath))).click()
        html_source = browser.page_source
        soup = BeautifulSoup(html_source, 'html5lib')
        litige_img_list = soup.findAll(attrs={'class':'litige',
                                              'id':re.compile("^tr1")})

        for j, element in enumerate(litige_img_list):

            #id_url = re.compile(r'\"\/IEL_orgaInscriptionCoureur.php\?(.*)\"').findall(str(element))[0]
            id = element.attrs.get('id').split('_')[-1]
            img_url = re.compile(r'\'(https[\s\S]*?)\'').findall(str(element))

            for k, i_url in enumerate(img_url):
                if l_duo:
                    l_part = k + 1

                filename = i_url.split("/")[-1]
                print("%d:%s"%(j,filename))

                credentials = filename
                # do not take extension and year, that's y -2
                entry = extract_name(credentials)[:-2]
                entry_found = fh.lookup(entry)

                if not entry_found:
                    src = os.path.join(TEMP_PATH, filename)
                    i_url = i_url.replace(' ','%20')
                    urlretrieve(i_url, src)

                    try:
                        if not filename.endswith(".pdf"):
                            src_pdf = TEMP_PATH +'/'+ filename.split('.')[0] + ".pdf"

                            if not os.path.isfile(src_pdf):
                                # convert other ext files to pdf if not found in DIR_PATH_pdf
                                im_temp = Image.open(src).convert('L')
                                # crop white space
                                #im_temp = trim(im_temp)
                                #im_temp = rotation(im_temp)
                                im_temp.save(src_pdf, "pdf", optimize=True, quality=85)

                            src = src_pdf

                        PyPDF2.PdfFileReader(open(src, "rb"))
                        img = convert_from_path(src, fmt="png")[0].convert('L')

                        option = 0
                        img2 = img.copy()
                        score_total = pd.DataFrame(data=[[0,0,0,0]],
                                                   columns=fh.df.columns.tolist()[6:],
                                                   index = [i])

                        while((score_total.sum(axis=1).values != 4) and (option < 6)):
                            img, img2 = thresholding(img, img2, DIR_PATH, option)
                            im = trim(img2)
                            txt_img = pytesseract.image_to_string(im)
                            txt = text_preprocess(txt_img)

                            if verbose:
                                print("try ", option + 1,":", txt)

                            temp = keyword_lookup(i, fh.df, filename, txt,
                                                  BEGIN_DATE, C_KEYWORDS,
                                                  l_prod=True)
                            score = temp.iloc[:,6:]*1
                            score_total += score

                            if verbose:
                                print(score_total)

                            score_total.replace(2, 1, inplace=True)
                            option += 1


                        # Removing temporary pdf file
                        for f in score_total.columns:
                            if(int(score_total[f][:1])==1):
                                temp[f][:1]=True
                            else:
                                temp[f][:1]=False

                        score = temp[keep_columns].sum(axis=1).values
                        print("Score:%d" %score)

                        if score!=4:
                            # keep only cases that are different from 4 for
                            # improvement purpose
                            fh.add(temp)

                        if verbose:
                            print(fh.df)

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
                            columns=fh.df.columns,
                            index=[j])
                        fh.add(temp)
                        print("invalid PDF file")
                else:
                    print('Certificat déjà examiné')

        fh.delete()
        fh.save()

    browser.quit()
    shutil.rmtree(TEMP_PATH)
