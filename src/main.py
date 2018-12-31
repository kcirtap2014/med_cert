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
from helper_functions import (text_preprocess, trim, keyword_lookup,
                              feature_engineering, find_cluster,
                              horizontal_clustering)
from image_preprocessing import ImagePreprocessing
from skimage.transform import resize
from collections import defaultdict
from joblib import load
from filepaths import FilePaths


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
    char_acc_file = FilePaths.fnAccuracy
    char_list = FilePaths.fnCharList
    retained_file_list = os.path.join(RETAINED_FILE_PATH,'retained_file_score_0')
    decoderType = DecoderType.WordBeamSearch
    #keywords_preprocessed = []
    # sort by alphabetical order
    sorted_file_list = sorted(file_list)

    with open(retained_file_list,'rb') as fp:
        sorted_file_list = pickle.load(fp)

    # import model
    model = Model(open(char_list).read(), decoderType, mustRestore=True)
    print("Model loaded")
    kmeans = load(os.path.join(MODEL_PATH,'kmeans.joblib'))
    print("Kmeans for text/manuscript distinction loaded")
    clf = load(os.path.join(MODEL_PATH,'clf_lr.joblib'))
    print("SVM for text/manuscript distinction loaded")

    print(open(char_acc_file).read())

    for i, file in enumerate(sorted_file_list[0:1]):
        # some default dicts to store information
        cert_features = []
        resize_images = []
        dimensions = []
        X_cert = []

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
            #img = cv2.imread(src_pdf)
            img = convert_from_path(src_pdf, fmt="png")[0]#.convert('L')
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
        else:
            img = convert_from_path(src, fmt="png")[0]#.convert('L')
            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

        # resize image
        print(np.shape(img))
        img = cv2.resize(img, A4_100DPI)
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
            txt = ""
            resizer = Resize(Model.imgSize)
            segmentation = Segmentation(img, p_hough = True)
            segmentation.run()
            seg_img = segmentation.image
            img_new_comp_ = segmentation.new_components_
            print("Num components:", len(img_new_comp_))

            for i, (dim, segment) in enumerate(img_new_comp_):
                sub_image = segment

                if not sub_image.size==0:
                    resize_img = resizer.transform(sub_image)
                    output = feature_engineering(resize_img, l_daisy=False,
                                                 l_hog=False)

                    if not output[0] is None:
                        cert_features.append(output[0])
                        resize_images.append(resize_img)
                        dimensions.append(dim)

            # create a sorted index based on dimensions
            index_cert = horizontal_clustering(dimensions)

            for features in cert_features:
                bovw_feature_cert = find_cluster(kmeans, features)
                X_cert.append(bovw_feature_cert)

            y_pred_cert = clf.predict(np.array(X_cert))
            # mapping
            index_hand = [index_cert[i] for i, pred in enumerate(y_pred_cert)]# if pred==False ]

            infer_resize_images = list(np.array(resize_images)[index_hand])
            #index_hand = index_cert
            print("Num handwritten words:%d" %(len(index_hand)))

            freq = Model.batchSize
            # add black image to compete the batch
            infer_resize_images += list(np.zeros([freq - len(infer_resize_images)%freq,
                                            Model.imgSize[0], Model.imgSize[1]]))
            current_id = 0
            current_turn = 0
            n_images = len(infer_resize_images)
            n_turn = int(n_images/Model.batchSize)

            while current_turn < n_turn:

                batch = Batch(None, np.array(infer_resize_images)[current_id:freq])
                # fill all batch elements with same
                recognized = model.inferBatch(batch) # recognize text
                txt += " ".join(recognized)

                current_id = freq
                current_turn += 1
                freq += freq

            #print('Recognized:', '"' + recognized[0] + '"') # all batch elements  hold same result
            if False: #len(recognized[0])>2:
                fig,ax = plt.subplots(2,1)
                ax[0].imshow(sub_image,cmap="gray")
                ax[1].imshow(resize_img,cmap="gray")
                plt.show()

            print(txt)
            pdb.set_trace()

        else:
            segmentation = Segmentation(img, p_hough = True)
            segmentation.run()
            seg_img = segmentation.image
            img_new_comp_ = segmentation.new_components_

            (wt,ht) = Model.imgSize
            resizer = Resize(Model.imgSize)

            for i, (x,y,w,h) in enumerate(img_new_comp_):
                sub_image = seg_img[y:y+h, x:x+w]

                if not sub_image.size==0:
                    resize_img = resizer.transform(sub_image)
                    #resize_img = resize(sub_image, (wt,ht),
                    #                        anti_aliasing=True)
                    output = feature_engineering(resize_img, l_daisy=False,
                                                 l_hog=False)

                    if not output[0] is None:
                        cert_features[i] = output[0]
                        resize_images[i] = resize_img


                    batch = Batch(None, [resize_img] * Model.batchSize) # fill all batch elements with same
                    recognized = model.inferBatch(batch) # recognize text
                    print('Recognized:', '"' + recognized[0] + '"') # all batch elements  hold same result
                    if False:
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
