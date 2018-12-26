from urllib.request import urlretrieve, urlopen
from bs4 import BeautifulSoup
import os
import re
import shutil
import pdb
import pandas as pd
import numpy as np
import warnings
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from helper_functions import (extract_name, find_match)
from config import dir_path, chrome_driver_path

warnings.filterwarnings("ignore", category=UserWarning)
temp_path = dir_path + "/temp_file/"
download_path = dir_path +"/list2018/"
output_path = dir_path +"/output/"


# check if fig_path exists
if not os.path.exists(temp_path):
    os.makedirs(temp_path)

if __name__ == "__main__":
    #set up dataframe
    #url = 'https://www.topchrono.biz/organisateur/index.html'
    #values = {'login': 'inscriptionstvalentin@frontrunnersparis.org',
    #          'pwd': '2017STVALMDP'}
    #url2= 'https://www.topchrono.biz/espace-organisateur.php?login=Logstvalen2019!&pwd=Passtvalen2019!&redirect=espace-organisateur.php'

    # some flags
    #timeout = 3
    l_duo = False

    # set up browser
    #browser = webdriver.Chrome(chrome_driver_path)
    #browser.get(url)
    # login
    #browser.find_element_by_name('login').send_keys(values['login'])
    #browser.find_element_by_name('pwd').send_keys(values['pwd'])
    #inscr_btn = WebDriverWait(browser, timeout).until(EC.presence_of_element_located((By.NAME, 'connecter')))
    #inscr_btn.click()
    #browser.find_element_by_name('connecter').click()
    #browser.get(url2)
    #_ = session_requests.get(url2, headers = dict(referer = url2))
    #fchar = np.arange(7,10)

    #url = 'https://www.topchrono.biz/org_tele_insc.php'
    #urlretrieve(url, download_path + 'inscriptions_cur.xls' )
    #list_cur_year = download_path + "inscriptions_cur.xls"
    #df_cur_year = pd.read_excel(list_cur_year)
    for i in range(3):
        if i==2:
            #only activate when it's for duo category
            l_duo = True

        list_last_year = download_path + "inscriptions_%d.xls" %(i+1)
        list_cur_year = download_path + "inscriptions_cur_%d.xls" %(i+1)

        df_last_year = pd.read_excel(list_last_year)
        df_cur_year = pd.read_excel(list_cur_year)
    
        df_savename = 'df_%d' %(i+1)
        df = pd.DataFrame(columns=["nom", "prénom", "présent", "error"])
        #for j, element in enumerate(litige_patrick):#litige_img_list[-1:]):
        for j, element in df_cur_year.iterrows():
            if l_duo:
                for i in range(1,3):
                    keep_columns = ['Nom coureur %d' %(i), 'Prénom coureur %d' %(i)]
                    credentials = element[keep_columns].values
                    valid_registration, error = find_match(credentials, df_last_year, i)
                    df_temp = pd.DataFrame([list(credentials) + [valid_registration, error]],
                                            columns=["nom", "prénom", "présent", "error"],
                                            index=[j])
                    df = df.append(df_temp)

            else:
                keep_columns = ['Nom', 'Prénom']
                credentials = element[keep_columns].values
                valid_registration, error = find_match(credentials, df_last_year, 0)
                df_temp = pd.DataFrame([list(credentials) + [valid_registration, error]],
                                        columns=["nom", "prénom", "présent", "error"],
                                        index=[j])
                df = df.append(df_temp)

        df.to_csv(output_path + df_savename + ".csv")

    #browser.quit()
    shutil.rmtree(temp_path)
