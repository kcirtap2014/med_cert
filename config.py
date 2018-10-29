# config files
# some flags
import os
import pandas as pd

dir_path = os.getcwd()

# load certificates
cert_dir = dir_path + '/TestCertificats/'

# prepare an empty DataFrame
df_exp = pd.DataFrame(columns=[
      "Nom", "Prenom", "Date", "Ext", "FileName", "C_Nom",
      "C_Prenom", "C_Date", "C_Mention"])

# define file_list
ext = tuple(['pdf', 'jpg', 'jpeg', 'png'])
file_list = [f for f in os.listdir(cert_dir) if f.endswith(ext)]
c_keywords = [("athle", "running", "course", "pied", "compétition",
                 "athlétisme", "semi-marathon", "marathon")]
begin_date = '17/2/2018'
