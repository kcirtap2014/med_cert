# config files
# some flags
import os
import pandas as pd

DIR_PATH = os.getcwd()

# load certificates
cert_dir = DIR_PATH + '/TestCertificats/'
CHROME_DRIVER_PATH = '/Users/pmlee/Documents/chrome/chromedriver'

# prepare an empty DataFrame
df_exp = pd.DataFrame(columns=[
      "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom",
      "C_Prenom", "C_Date", "C_Mention"])

df_prod = pd.DataFrame(columns=[
      "R_Date","Nom", "Prenom", "Year", "Ext", "FileName", "C_Nom",
      "C_Prenom", "C_Date", "C_Mention"])

# define file_list
ext = tuple(['pdf', 'jpg', 'jpeg', 'png'])
file_list = [f for f in os.listdir(cert_dir) if f.endswith(ext)]
C_KEYWORDS = [("athle", "running", "course", "pied", "compétition",
                 "athlétisme", "semi-marathon", "marathon")]
BEGIN_DATE = '17/2/2018'