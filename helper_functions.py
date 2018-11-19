import math
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageChops
from skimage.feature import daisy, hog
import re
from sklearn.cluster import MiniBatchKMeans
from skimage import filters
from skimage.filters import threshold_local
import editdistance
import pdb
from unidecode import unidecode

def extract_name(filename):
    """
    extracts name from filename. filenames are stored as follows:
    surname_firstname_valid-date.extension

    Parameters:
    -----------
    filename: str

    Returns:
    --------
    keywords: array_like
        a list of keywords for identification
    """

    keywords = re.split('_|\.', filename)

    return keywords


def find_whole_word(w):
    """
    look for whole word only

    Parameters:
    -----------
    word: str

    Returns:
    --------
    regex search function
    """

    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def find_partial(w, txt, n_keep_char):
    """
    apply editdistance to groud truth words and extracted text. This also
    works when there are tuples such as composed name ("Jean","Noel"). For now,
    we tolerate an distance of <=2.

    Parameters:
    -----------
    w: tuple of strings or string
        ground truth

    txt: string
        extraceted text

    found: Boolean
        True if Levenshtein distance <=2

    n_keep_char: int
        number of characters to keep, to avoid comparison with all extracted
        words. Default: 3

    Returns:
    --------
    found: boolean
    """
    found = False

    if isinstance(w, tuple):
        # for composed words, search only for the fist word in the bigram tuple
        p = re.compile(
            r'\b({0})\b'.format(w[0][:n_keep_char]), flags=re.IGNORECASE)
        dict_wordlist = create_ngram(txt)
        matched_list = [(i, bool(p.search(wtxt[0][:n_keep_char])))
                        for i, wtxt in enumerate(dict_wordlist)
                        if bool(p.search(wtxt[0][:n_keep_char]))]

        found_list = np.zeros(len(w))

        for key, value in matched_list:
            wordlist = dict_wordlist[key]

            for i, word in enumerate(wordlist):
                if editdistance.eval(word, w[i]) <= 3:
                    #print("Found", word, w[i])
                    found_list[i] = True

        if np.sum(found_list) == len(w):
            found = True

    else:
        p = re.compile(
            r'\b({0})\b'.format(w[:n_keep_char]), flags=re.IGNORECASE)
        dict_wordlist = [wtxt for wtxt in txt.split(" ")]
        matched_list = [(i, bool(p.search(wtxt[:n_keep_char])))
                        for i, wtxt in enumerate(dict_wordlist)
                        if bool(p.search(wtxt[:n_keep_char]))]

        for key, value in matched_list:
            word = dict_wordlist[key]

            if editdistance.eval(word, w) <= 3:
                found = True

    return found

def thresholding(img, option=0):

    if option == 1:
        IMG0 = np.array(img)
        thresh = cv2.adaptiveThreshold(IMG0,255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,15,11)
        img_thresh = Image.fromarray(thresh)
        im = trim(img_thresh)

        # pytesseract will sometimes crash if the image is too big (but only after thresholding for some reason)
        if (np.array(img).shape[0]*np.array(img).shape[1] > 80000000):
            im.thumbnail((2000,2000),Image.ANTIALIAS)

    elif option == 2:
        img.thumbnail((1000,1000))
        im = trim(img)

    elif option == 3:
        img.thumbnail((1000,1000),Image.ANTIALIAS)
        im = trim(img)

    elif option == 4:
        IMG0=np.array(img)
        thresh=cv.adaptiveThreshold(IMG0,255,
                                    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv.THRESH_BINARY,15,11)
        img_thresh=Image.fromarray(thresh)
        im = trim(img_thresh)

    else:
        im = img
    mat_img = np.asarray(im)

    return mat_img

def find_match(credentials, df, l_part):
    """
    find match in the input df
    Parameters:
    -----------
    credentials: dict

    """
    found = False
    error = 0

    if l_part>0:
        cond = np.logical_and.reduce(
        (df['Nom coureur %d' %l_part]==credentials[0].upper(),
         df['Prénom coureur %d' %l_part]==credentials[1].upper()))

        new_df = df[cond]

        if len(new_df)==0:
            if l_part==1:
                l_part+=1

            elif l_part==2:
                l_part-=1

            cond = np.logical_and.reduce(
                (df['Nom coureur %d' %l_part]==credentials[0].upper(),
                 df['Prénom coureur %d' %l_part]==credentials[1].upper()))

            new_df = df[cond]
            error = 2

    else:
        cond = np.logical_and.reduce((df['Nom']==credentials[0].upper(),
                          df['Prénom']==credentials[1].upper()))
                          #df["Année de naissance"]==int(credentials[2])))

        new_df = df[cond]

        if len(new_df)==0:
            cond  = np.logical_and.reduce((
                    df['Prénom']==credentials[0].upper(),
                    df['Nom']==credentials[1].upper()))
                    #df["Année de naissance"]==int(credentials[2])))

            new_df = df[cond]
            error = 1

    if len(new_df) == 1:
        found = True
        print("Identifiant trouvé")

    elif len(new_df)>1:
        print("Plusieurs identifiants du même nom ont été trouvés")

    else:
        print("Identifant non-trouvé")

    return found, error

def keyword_lookup(current_id, df, credentials, txt, begin_date, c_keywords,
                   n_keep_char=2, l_prod=False):
    """
    look up the keywords from extracted words of the document

    Parameters:
    -----------
    current_id: int
        index for pandas DataFrame indexation

    df: pandas dataframe
        result dataframe

    filename: str

    txt: str
        extracted texts of the document

    begin_date: str
        begin date

    c_keywords: list of tuples
        tuples of keywords

    n_keep_char: int
        number of first letters to be kept

    Returns:
    --------
    df: pandas dataframe
        result dataframe
    """

    if l_prod:
        extracted = extract_name(credentials)
    else:
        extract_from_file = extract_name(credentials)
        extracted = extract_from_file
    # (athle running) for license FFA
    keywords = extracted[1:-1] + c_keywords

    keywords_preprocessed = []

    # prprocess keywords, special processing for tuples, make everything
    # a tuple for easier processing later
    for word in keywords:
        tuple_keywords = []

        if isinstance(word, tuple):
            for t in word:
                tuple_keywords.append(text_preprocess(t))

        else:
            tuple_keywords.append(text_preprocess(word))

        keywords_preprocessed.append(tuple(tuple_keywords))

    temp_df = pd.DataFrame(
        [extracted + [credentials, False, False, False, False]],
        columns=df.columns,
        index=[current_id])

    df = df.append(temp_df)
    dates = parse_date(txt)

    # retain only columns for tick boxes
    cols = df.columns[6:]
    found = np.zeros(4) # 4 columns to check

    for i, keywords in enumerate(keywords_preprocessed):
        found_temp = []

        for key in keywords:
            if i == 2:
                #check for dates, condition of date match has to be modified for
                # production
                begin_date = pd.to_datetime(begin_date, dayfirst=True)

                # we don't need to specify end date, if the begin time is more
                # than 1 year before the course is supposed to take place,
                # that should be fine
                end_date = pd.to_datetime(begin_date) + pd.Timedelta(weeks=52)

                for date in dates:
                    try:
                        cur_date = pd.to_datetime(date, dayfirst=True)
                        #date_match = (pd.to_datetime(date) == pd.to_datetime(key))
                        #date_match = np.logical_and(cur_date >= begin_date,
                        #                            cur_date <= end_date)
                        date_match = np.logical_and(cur_date >= begin_date, cur_date<=end_date)
                    except ValueError:
                        date_match = False

                    found_temp.append(date_match)

            else:
                # if we have the exactly the same word, consider True,
                # otherwise proceed with Levenshtein distance evaluation

                found_it = bool(find_whole_word(key)(txt))

                if not found_it:
                    composed_key = None
                    # consider words that start with the same 3 letters
                    if (bool(
                            re.compile(
                                r'[a-z]*\-[a-z]*'.format(key),
                                flags=re.IGNORECASE).search(key))):
                        key_temp = key.replace("-", " ")
                        composed_key = create_ngram(key_temp)[0]

                    if composed_key is not None:
                        key_partial = composed_key
                    else:
                        key_partial = key

                    found_it = find_partial(key_partial, txt, n_keep_char)

                found_temp.append(found_it)

        if i==3:
            # found conditions

            found_temp = np.array(found_temp)

            found[i] = np.logical_or.reduce((
                        np.sum(found_temp[:2]) == len(found_temp[:2]),
                        np.sum(found_temp[2:5]) == len(found_temp[2:5]),
                        np.sum(found_temp[[4,6]]) == len(found_temp[[4,6]]),
                        np.sum(found_temp[[4,7]]) == len(found_temp[[4,7]]),
                        np.sum(found_temp[[4,5]]) == len(found_temp[[4,5]])))

        else:
            # for date, if we have one or more matches, that is valid
            found[i] = (np.sum(found_temp) >= len(keywords))

        df.at[current_id, cols[i]] = bool(found[i])

    # return only the latest one
    return df[-1:]


def replace_month(date):
    """
    convert month in letters to number

    Parameters:
    -----------
    date: string
        date with month written in letters


    Returns:
    --------
    std_date: string
        date writen in this form: day/month/date
    """

    month_name = {'janv': 1, 'fevr': 2, 'mars': 3, 'avri': 4, 'mai': 5,
                  'juin': 6, 'juil': 7, 'août': 8, 'aout':8, 'sept': 9, 'octo': 10,
                  'nove': 11,'dece': 12
                }
    try:
        date = re.sub(r"(\d)([a-z])", r"\1 \2", date)
        day, month, year = date.split(' ')

        # take the first 4 letters
        month = month[:4]
        std_date = str(day)+ "/" + str(month_name[month])+ "/" +str(year)
    except (ValueError, KeyError):
        std_date = ""

    return std_date

def rotation(im):
    """
    rotate image if height < width

    Parameters:
    -----------
    im: PIL image

    Return:
    -------
    resulting image
    """

    height, width = np.shape(im)

    if height < width:
        im = im.rotate(-90, expand=True)

    return im

def create_ngram(txt, n=2):
    """
    create n_grams

    Parameters:
    -----------
    txt: string

    n: int
        for bigram, n=2 (default value)

    Returns:
    --------
    list of ngrams
    """

    return list(zip(*[txt.split()[i:] for i in range(n)]))

def parse_date(txt):
    """
    look for dates in txt

    Parameters:
    -----------
    txt: string
        extracted text

    txt_arr: array_like
        list of dates
    """

     # sometimes 1 can be detected as l, or spaces, we take that into account
    date_arr1 = re.compile(
        r"(?:l\d{1,2}|\d{1,2}l|ll|\d{1,2})[\/\-\s\,]{1,2}" \
        + r"(?:l\d{1,2}|\d{1,2}l|ll|\d{1,2})[\/\-\s\,]{1,2}" \
        + r"(?:l\d{1}|\d{1}l|ll|\d{2}l\d{1}|\d{2,4})"
    ).findall(txt)

    # convert l to 1 if applicable
    date_arr1 = [date.replace("l","1") for date in date_arr1]

    # get rid of space
    date_arr1 = [date.replace(" ","") for date in date_arr1]

    # replace comma by slash
    date_arr1 = [date.replace(",","/") for date in date_arr1]

    date_arr2_temp = re.compile(
        r"\d{1,2}[a-z]*[\.\-\s\/]?" \
        + r"(?:jan|fev|mar|avr|mai|juin|juil|ao[a-z]{1,}|sept|oct|nov|dec)" \
        + r"(?:\.|[a-z])*[\,\s\-\/]{0,2}\d{2,4}",
        flags=re.IGNORECASE).findall(txt)

    # convert erroneous aout writing to the correct one
    date_arr2_temp = [re.sub(r"ao[a-z]{1,}", "aout", date)
                      for date in date_arr2_temp]

    date_arr2 = [replace_month(date) for date in date_arr2_temp]
    # concatenate both lists
    date_arr = date_arr1 + date_arr2
    # print(date_arr)

    return date_arr


def text_preprocess(txt):
    """
    text preprocessing: lowercase, get rid of punctuations, and convert \n to \s

    Parameters:
    -----------
    txt: string
        extracted text

    txt_skipline: string
        parsed text
    """

    # lowercase
    txt_small_case = txt.lower()
    # get rid of punctuations
    txt_punct = re.compile(r'<.*?>|[^\w\s+\.\-\#\+]\/').sub('', txt_small_case)
    txt_skipline = re.compile(r'\n').sub(' ', txt_punct)
    # replace em-dash
    txt_dashline = re.compile(r'\u2014').sub('-', txt_skipline)
    # remove accent
    txt_accent_free = unidecode(txt_dashline)

    return txt_accent_free

def trim(im):
    """
    crop white space of the image

    Parameters:
    -----------
    im: PIL image
        input image

    Returns:
    --------
    cropped image
    """
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)

def local_thresholding(mat_img, block_size=35, offset=10):
    """
    local thresholding

    Parameters:
    -----------
    mat_img: arrays of uint8
        img array

    block_size: int
        window size. Default value:35

    offset: int

    Returns:
    --------
    mat_mask_img: arrays of uint8
        array of masked image

    mask_img: PIL image
        resulting image
    """

    local_thresh = threshold_local(mat_img, block_size, offset=offset)
    mask = mat_img>local_thresh
    mat_mask_img = mat_img*mask
    mask_img = Image.fromarray(np.uint8(255*mat_mask_img))

    return mat_mask_img, mask_img

def wordSegmentation(img, kernelSize=25, sigma=11, theta=7, minArea=0):
    """Scale space technique for word segmentation proposed by R. Manmatha: http://ciir.cs.umass.edu/pubfiles/mm-27.pdf

    Args:
        img: grayscale uint8 image of the text-line to be segmented.
        kernelSize: size of filter kernel, must be an odd integer.
        sigma: standard deviation of Gaussian function used for filter kernel.
        theta: approximated width/height ratio of words, filter function is distorted by this factor.
        minArea: ignore word candidates smaller than specified area.

    Returns:
        List of tuples. Each tuple contains the bounding box and the image of the segmented word.
    """

    # apply filter kernel
    kernel = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgThres = 255 - imgThres

    # find connected components. OpenCV: return type differs between OpenCV2 and 3
    if cv2.__version__.startswith('3.'):
        (_, components, _) = cv2.findContours(imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        (components, _) = cv2.findContours(imgThres, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # append components to result
    res = []
    for c in components:
        # skip small word candidates
        if cv2.contourArea(c) < minArea:
            continue
        # append bounding box and image of word to result list
        currBox = cv2.boundingRect(c) # returns (x, y, w, h)
        (x, y, w, h) = currBox
        currImg = img[y:y+h, x:x+w]
        res.append((currBox, currImg))

    # return list of words, sorted by x-coordinate
    return sorted(res, key=lambda entry:entry[0][0])


def prepareImg(img, height):
    """
    convert given image to grayscale image (if needed)
    and resize to desired height
    """
    assert img.ndim in (2, 3)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = img.shape[0]
    factor = height / h
    return cv2.resize(img, dsize=None, fx=factor, fy=factor)


def createKernel(kernelSize, sigma, theta):
    """create anisotropic filter kernel according to given parameters"""
    assert kernelSize % 2 # must be odd size
    halfSize = kernelSize // 2

    kernel = np.zeros([kernelSize, kernelSize])
    sigmaX = sigma
    sigmaY = sigma * theta

    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfSize
            y = j - halfSize

            expTerm = np.exp(-x**2 / (2 * sigmaX) - y**2 / (2 * sigmaY))
            xTerm = (x**2 - sigmaX**2) / (2 * math.pi * sigmaX**5 * sigmaY)
            yTerm = (y**2 - sigmaY**2) / (2 * math.pi * sigmaY**5 * sigmaX)

            kernel[i, j] = (xTerm + yTerm) * expTerm

    kernel = kernel / np.sum(kernel)
    return kernel

def clustering(features, n_cluster):
    """
    perform clustering using MiniBatchKMeans

    Parameters:
    -----------
    features: array_like
        array of features

    n_cluster: int
        number of clusters

    Returns:
    --------
    kmeans: scikit learn model
    """
    kmeans = MiniBatchKMeans(n_cluster, batch_size=n_cluster * 10)
    kmeans.fit(features)

    return kmeans

def find_cluster(cluster_model, features):
    """
    find to which clusters each feature belongs

    Parameters:
    -----------
    cluster_model: scikit-learn model

    features: array_like
        array of features
    """
    img_clusters = cluster_model.predict(features)
    cluster_freq_counts = pd.DataFrame(
        img_clusters, columns=['cnt'])['cnt'].value_counts()
    bovw_vector = np.zeros(cluster_model.n_clusters)

    ##feature vector of size as the total number of clusters
    for key in cluster_freq_counts.keys():
        bovw_vector[key] = cluster_freq_counts[key]

    bovw_feature = bovw_vector / np.linalg.norm(bovw_vector)

    return list(bovw_feature)

def feature_engineering(img, step=32, radius=32, histograms=8, orientations=8,
                        visualize=False, l_hog=True, l_daisy=True, l_sift=True):
    """
    feature engineering with HOG, DAISY and/or SIFT descriptors

    Parameters:
    -----------
    img: input image

    step: int
        daisy descriptor parameter. it defines the step between descriptors

    radius: int
        daisy descriptor parameter. it defines the radius of the descriptor

    histograms: int
        daisy descriptor parameter. number of histograms per descriptor

    orientations: int
        daisy descriptor parameter. number of orientations per descriptor. each
        orientation is 45°

    visualize: boolean
        true if want to return image

    l_hog: boolean
        true if use HOG descriptor

    l_daisy: boolean
        true if use DAISY descriptor

    l_sift: boolean
        true if use SIFT descriptor

    Return:
    -------
    feature descriptors
    """

    #mat_img_filter = image_preprocessing(img)
    mat_img = np.array(img)
    mat_img_filter = filters.median(mat_img)
    output = []

    if l_hog:
        if visualize:
            fd, img_hog = hog(mat_img_filter, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(1, 1), visualize=visualize,
                              feature_vector=True)
            output_hog = fd, img_hog

        else:
            fd = hog(mat_img_filter, orientations=8, pixels_per_cell=(16, 16),
                     cells_per_block=(1, 1), visualize=visualize,
                     feature_vector=True)
            output_hog = fd

        output.append(output_hog)

    if l_daisy:
        # apply daisy feature extraction
        if visualize:
            descs, img_daisy = daisy(mat_img_filter, step=step, rings=2,
                                     histograms=histograms, radius=radius,
                                     normalization='l2',
                                     orientations=orientations,
                                     visualize=visualize)

            output_daisy = descs, img_daisy

        else:
            descs = daisy(mat_img_filter, step=step, rings=2, radius=radius,
                          histograms=histograms, normalization='l2',
                          orientations=orientations, visualize=visualize)

            descs_num = descs.shape[0] * descs.shape[1]
            daisy_descriptors = descs.reshape(descs_num, descs.shape[2])
            output_daisy = daisy_descriptors

        output.append(output_daisy)

    if l_sift:

        sift = cv2.xfeatures2d.SIFT_create()
        # convert to gray scale
        #img_gray = cv2.cvtColor(mat_img, cv2.COLOR_BGR2GRAY)

        # denoise
        #median_blur_img = cv2.medianBlur(img_gray, ksize=1)

        # equalizer: contrast adjustment
        img_eq = cv2.equalizeHist(img)

        kp, descs = sift.detectAndCompute(img_eq, None)
        output_sift = descs

        if visualize:
            img_sift = cv2.drawKeypoints(img_eq, kp, None,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            output_sift = descs, img_sift

        output.append(output_sift)

    return output

def image_preprocessing(img):
    # crop white space
    #im = trim(img)
    mat_img = np.asarray(img)
    # get rid of salt and pepper noise
    # convert to gray scale as only the luminosity is important

    # denoise
    mat_img_filter = filters.median(mat_img)

    # contast adjustment

    return mat_img_filter
