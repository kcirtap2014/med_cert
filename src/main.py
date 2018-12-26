import pandas as pd
import seaborn as sns
import numpy as np
import pytesseract
import os
import argparse
from PIL import Image
from pdf2image import convert_from_path
import warnings
import pickle
import cv2
import pdb
import matplotlib.pylab as plt
from DataLoader import Batch
from Model import Model, DecoderType
from config import (DIR_PATH, df_exp, file_list, C_KEYWORDS, BEGIN_DATE,
                    PDF_PATH, CERT_PATH, MODEL_PATH,RETAINED_FILE_PATH, A4_100DPI)
from resize import Resize
from segmentation import Segmentation
from helper_functions import (text_preprocess, trim, keyword_lookup)
from image_preprocessing import ImagePreprocessing
from skimage.transform import resize

#### To use directly in python , set working directory to contain helper_functions.py and the directory containing certificates
#### os.chdir("Desktop/Certificats/Cert_Recognition/")

### To use pytesseract after standard windows installation:
### pytesseract.pytesseract.tesseract_cmd='C:/Program Files (x86)/Tesseract-OCR/tesseract'

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_style("ticks")
parser = argparse.ArgumentParser()
parser.add_argument("--verbose", help="Activate verbose",type=bool, nargs='?',
                    const=True, default=False)
args = parser.parse_args()

if __name__ == '__main__':
    verbose  = bool(args.verbose)

    # some flags
    char_acc_file = os.path.join(MODEL_PATH,'accuracy.txt')
    char_list= os.path.join(MODEL_PATH,'charList.txt')
    retained_file_list = os.path.join(RETAINED_FILE_PATH,'retained_file_score_0')
    decoderType = DecoderType.WordBeamSearch
    #keywords_preprocessed = []
    # sort by alphabetical order
    sorted_file_list = sorted(file_list)
    with open(retained_file_list,'rb') as fp:
        sorted_file_list = pickle.load(fp)

    # import model
    model = Model(open(char_list).read(), decoderType,
                       mustRestore=True)
    print(open(char_acc_file).read())

    for i, file in enumerate(sorted_file_list[5:6]):
        filename = os.fsdecode(file)
        src = os.path.join(CERT_PATH, filename)
        print("%d:%s"%(i,filename))

        if not filename.endswith(".pdf"):
            src_pdf = PDF_PATH + filename.split('.')[0] + ".pdf"

            if not os.path.isfile(src_pdf):
                # convert other ext files to pdf if not found in DIR_PATH_pdf
                im_temp = Image.open(src).convert('L')
                # crop white space
                im_temp.save(src_pdf, "pdf", optimize=True, quality=85)

            img = convert_from_path(src_pdf, fmt="png")[0].convert('L')
        else:
            img = convert_from_path(src, fmt="png")[0].convert('L')
        # resize image
        print(np.shape(img))
        img = cv2.resize(np.asarray(img), A4_100DPI)
        print("after", np.shape(img))

        # Setting variables for the loop
        option = 0
        #img2 = copy(img)
        score_total = pd.DataFrame(data=[[0,0,0,0]],
                                   columns=df_exp.columns.tolist()[5:9],
                                   index = [i])
        if False:
            while((score_total.sum(axis=1).values != 4) and (option < 2)):
                im_proc = ImagePreprocessing(img, verbose=verbose, graph_line=False,    morphology=False, option=option)
                im_proc.process()
                txt_img = pytesseract.image_to_string(im_proc.image)
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

        l_segmentation = True

        if l_segmentation:
            resizer = Resize(Model.imgSize)
            segmentation = Segmentation(img, p_hough = True)
            segmentation.run()
            seg_img = segmentation.image
            img_new_comp_ = segmentation.new_components_

            for dim, segment in img_new_comp_:
                sub_image = segment
                if not sub_image.size==0:
                    resize_img = resizer.transform(sub_image)
                    #resize_img = resize(sub_image, (wt,ht),
                    #                        anti_aliasing=True)

                    batch = Batch(None, [resize_img] * Model.batchSize) # fill all batch elements with same
                    recognized = model.inferBatch(batch) # recognize text
                    print('Recognized:', '"' + recognized[0] + '"') # all batch elements  hold same result
                    if len(recognized[0])>2:
                        fig,ax = plt.subplots(2,1)
                        ax[0].imshow(sub_image,cmap="gray")
                        ax[1].imshow(resize_img,cmap="gray")
                        plt.show()
                    
        else:

            segmentation = Segmentation(img, p_hough = True)
            segmentation.run()
            seg_img = segmentation.image
            img_new_comp_ = segmentation.new_components_
            print("Num components:", len(img_new_comp_))
            (wt,ht) = Model.imgSize
            resizer = Resize(Model.imgSize)

            for i, (x,y,w,h) in enumerate(img_new_comp_):
                sub_image = seg_img[y:y+h, x:x+w]

                if not sub_image.size==0:
                    resize_img = resizer.transform(sub_image)
                    #resize_img = resize(sub_image, (wt,ht),
                    #                        anti_aliasing=True)

                    batch = Batch(None, [resize_img] * Model.batchSize) # fill all batch elements with same
                    recognized = model.inferBatch(batch) # recognize text
                    print('Recognized:', '"' + recognized[0] + '"') # all batch elements  hold same result
                    if len(recognized[0])>2:
                        fig,ax = plt.subplots(2,1)
                        ax[0].imshow(sub_image,cmap="gray")
                        ax[1].imshow(resize_img,cmap="gray")
                        plt.show()
                        pdb.set_trace()
        pdb.set_trace()
        # Attempting to read text from images with various processing options
        if False:
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
        os.remove(DIR_PATH + "/temp.pdf")
    except:
        pass

    df_exp.to_csv(DIR_PATH + "/df_test3.csv", index=False)
