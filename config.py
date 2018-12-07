# config files
# some flags
import os
import pandas as pd

DIR_PATH = '/Users/pmlee/Documents/FRP/Cert_Recognition/Cert_Recognition'#os.getcwd()

# load certificates
CERT_PATH= os.path.join(DIR_PATH, 'TestCertificats')
CHROME_DRIVER_PATH = '/Users/pmlee/Documents/chrome/chromedriver'
OBS_PATH = os.path.join(os.path.dirname(DIR_PATH), 'nouvel-obs')
MODEL_PATH = os.path.join(os.path.dirname(DIR_PATH), 'model')
PDF_PATH = os.path.join(DIR_PATH, 'pdf')
TEMP_PATH = os.path.join(DIR_PATH, "temp_file")
OUTPUT_PATH = os.path.join(DIR_PATH, "output")

# prepare an empty DataFrame
df_exp = pd.DataFrame(columns=[
      "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom",
      "C_Prenom", "C_Date", "C_Mention"])

df_prod = pd.DataFrame(columns=[
      "R_Date","Nom", "Prenom", "Year", "Ext", "FileName", "C_Nom",
      "C_Prenom", "C_Date", "C_Mention"])

keep_columns = ["C_Nom","C_Prenom","C_Date","C_Mention"]
# define file_list
ext = tuple(['pdf', 'jpg', 'jpeg', 'png'])
file_list = [f for f in os.listdir(CERT_PATH) if f.endswith(ext)]
C_KEYWORDS = [("athle", "running", "course", "pied", "compétition",
                 "athlétisme", "semi-marathon", "marathon")]
BEGIN_DATE = '17/2/2018'
