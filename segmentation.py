import shutil
import pickle
import os
import numpy as np
import pytesseract
from PIL import Image
import matplotlib.pylab as plt
from pdf2image import convert_from_path
from skimage.filters import threshold_otsu
from helper_functions import (local_thresholding, wordSegmentation,
                              trim, thresholding, rotation, feature_engineering,
                              find_cluster)
import matplotlib.patches as patches
import argparse
import cv2
import pdb
from joblib import load
from collections import defaultdict
from config import DIR_PATH, MODEL_PATH, CERT_PATH, OBS_PATH, PDF_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--plot", help="Activate plot",type=bool, nargs='?',
                    const=True, default=False)
parser.add_argument("--obs", help="obs data",type=bool, nargs='?',
                    const=True, default=False)
args = parser.parse_args()

# run by typing python3 segmentation.py
if __name__ == '__main__':

    if bool(args.obs):
        INPUT_PATH = os.path.join(OBS_PATH, 'input')
        IMG_PATH = os.path.join(OBS_PATH,  'words')

        if os.path.exists(IMG_PATH):
            shutil.rmtree(IMG_PATH)

        elif not os.path.exists(IMG_PATH):
            # check if fig_path exists
            os.makedirs(IMG_PATH)

        file_list_crnn = os.listdir(INPUT_PATH)

    else:
        INPUT_PATH = os.path.join(DIR_PATH, 'input')
        IMG_PATH = os.path.join(DIR_PATH,'img_data')
        kmeans = load(os.path.join(MODEL_PATH,'kmeans.joblib'))
        clf_lr = load(os.path.join(MODEL_PATH,'clf_lr.joblib'))

        with open(os.path.join(DIR_PATH, 'retained_file_score_1'), 'rb') as fp:
            file_list_crnn = pickle.load(fp)

    sorted_file_list_crnn = sorted(file_list_crnn)

    for i, file in enumerate(sorted_file_list_crnn):
        filename = os.fsdecode(file)
        if bool(args.obs):
            src = os.path.join(INPUT_PATH, filename)
        else:
            src = os.path.join(CERT_PATH, filename)
        print("%d:%s"%(i,filename))

        if filename.endswith(".pdf"):
            # convert it to gray scale
            img = convert_from_path(src, fmt="png", dpi=200)[0].convert('L')

        else:
            #src_pdf = PDF_PATH + filename.split('.')[0] + ".pdf"
            # convert it to gray scale
            im_temp = Image.open(src).convert('L')

            img = rotation(im_temp)
            #im_temp.save(src_pdf, "pdf", optimize=True, quality=85)
            #img = convert_from_path(src_pdf, fmt="png")[0].convert('L')

        cropped_img = trim(img)
        #invert = cv2.bitwise_not(cropped_img)
        mat_img = np.asarray(cropped_img)

        #invert = cv2.bitwise_not(mat_img)
        #mat_img = thresholding(invert, option=0)
        bb_tuple = wordSegmentation(mat_img)#, kernelSize=55, sigma=211, minArea=500)

        # remove folder where images are stored
        #shutil.rmtree(IMG_PATH)
        #os.makedirs(IMG_PATH)
        rects = []

        if bool(args.obs):
            LOT_PATH  = os.path.join(IMG_PATH, 'lot%d' %i)
            os.makedirs(LOT_PATH)
        else:
            LOT_PATH = os.path.join(IMG_PATH, filename.split('.')[0])

        data = []

        for j, tup in enumerate(bb_tuple):
            x, y, w, h = tup[0]
            rect = patches.Rectangle((x,y),w,h,linewidth=1, edgecolor='r',
                                     facecolor='none')
            rects.append(rect)
            #img = tup[1]
            img = Image.fromarray(tup[1])
            SAVE_PATH = os.path.join(LOT_PATH, '%d_%d_img.png' %(x,y))

            if bool(args.obs):
                cv2.imwrite(SAVE_PATH, img)

            else:
                img = image_preprocessing(img)
                cert_features = defaultdict()
                output = feature_engineering(img, l_daisy=False, l_hog=False)

                if not output[0] is None:
                    cert_features[j] = output[0]
                    cv2.imwrite(SAVE_PATH, img)
                    data.append([j, SAVE_PATH, x, y, w, h])
            #pxmin = np.min(img)
            #pxmax = np.max(img)
            #imgContrast = (img - pxmin) / (pxmax - pxmin) * 255

            # increase line width
            #kernel = np.ones((3, 3), np.uint8)
            #img = cv2.erode(img, kernel, iterations = 1)
            #

            #txt = pytesseract.image_to_string(img)
        df_data = pd.DataFrame(data, columns = ["index","abs_path",
                                                "x", "y", "w", "h"])
        X_cert = []
        index_X_cert = []

        for key, features in cert_features.items():
            bovw_feature_cert = find_cluster(kmeans, features)
            X_cert.append(bovw_feature_cert)
            index_X_cert.append(key)

        y_pred_cert = clf_lr.predict(np.array(X_cert))

        df_pred = pd.DataFrame(list(zip(index_X_cert, y_pred_cert)),
                               columns = "index","y_pred" )

        df_merge = df_data.merge(df_pred, on="index").set_index("index")
        df_merge.to_csv(os.path.join(OUTPUT_PATH, '%s_df.csv' %filename.split(".")[0]))
        # I should probably add a dataframe
        if bool(args.plot):
            fig, ax = plt.subplots(figsize=(6,10))
            ax.imshow(mat_img, cmap='gray')

            for rect in rects:
                ax.add_patch(rect)
            plt.show()
