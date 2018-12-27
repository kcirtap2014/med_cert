import math
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageChops
from skimage.feature import (daisy, hog, canny)
import re
from sklearn.cluster import MiniBatchKMeans
from skimage import filters
from skimage.filters import threshold_local
import editdistance
import pdb
from unidecode import unidecode
from pdf2image import convert_from_path
from itertools import groupby
from config import DIR_PATH
from collections import defaultdict
from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage import draw
import matplotlib.pylab as plt

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

def jaccard_sim(s1,s2):

    inter = s1.intersection(s2)
    union = s1.union(s2)

    return len(inter)/len(union)

def find_partial(w, txt, n_keep_char, e_tolerance=3):
    """
    apply editdistance to groud truth words and extracted text. This also
    works when there are tuples such as composed name ("Jean","Noel"). For now,
    we tolerate a distance of <=2.

    Parameters:
    -----------
    w: tuple of strings or string
        ground truth

    txt: string
        extraceted text

    found: Boolean
        True if Levenshtein distance <=3

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
                if editdistance.eval(word, w[i]) < e_tolerance:
                    #print("Found", word, w[i])
                    found_list[i] = True

        if np.sum(found_list) == len(w):
            found = True

    else:
        # compare ngrams, different from previous approach where we only look at
        # the first 3 letters
        w_format = create_ngram(w, l_text=False, n=n_keep_char)
        w_ngram = w_format.split(" ")
        w_format = w_format.replace(" ","|")
        p = re.compile(
            r'\b({0})\b'.format(w_format), flags=re.IGNORECASE)
        # keep only words that are more than n_keep_char and get rid of
        # punctuations
        dict_wordlist = []

        for word in txt.split(" "):
            wout_punct = re.sub(r'[^\w\s]','', word)
            if len(wout_punct)>n_keep_char:
                dict_wordlist.append(wout_punct)

        matched_list = []

        for i, wtxt in enumerate(dict_wordlist):
            wtxt_ngrams = create_ngram(wtxt, l_text=False, n=n_keep_char)
            ngram_match = list(p.findall(wtxt_ngrams))

            # use Jaccard distance to perform the first filtering
            if jaccard_sim(set(ngram_match), set(w_ngram))>0.3:
                # the final decision is not made here, it is made by comparing
                # the Levenshtein distance below with the whole words, so it
                # is safe to let more words pass
                matched_list.append(i)

        for key in matched_list:
            word = dict_wordlist[key]

            if editdistance.eval(word, w) < e_tolerance:
                found = True

    return found

def thresholding(img, img2, DIR_PATH, option = 0):

    if option == 0:
        im = trim(img)
        mat_img = np.asarray(im)
        im = cv2.medianBlur(mat_img, 3)
        img2 = Image.fromarray(im)

    if option == 1:
        IMG0 = np.array(img)
        thresh = cv2.adaptiveThreshold(IMG0,255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,15,11)
        img2 = Image.fromarray(thresh)


        # pytesseract will sometimes crash if the image is too big (but only after thresholding for some reason)
        if (np.array(img).shape[0]*np.array(img).shape[1] > 80000000):
            img2.thumbnail((2000,2000),Image.ANTIALIAS)


    elif option == 2:
        img2.thumbnail((1000,1000))

    elif option == 3:
        img.thumbnail((1000,1000),Image.ANTIALIAS)
        img.save(DIR_PATH+"/temp.pdf")
        img = convert_from_path(DIR_PATH+"/temp.pdf", fmt="png")[0].convert('L')

    elif option == 4:
        IMG0=np.array(img)
        thresh=cv2.adaptiveThreshold(IMG0,255,
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY,15,11)
        img2=Image.fromarray(thresh)

    else:
        img2.thumbnail((1000,1000))

    return img, img2


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

    extracted = extract_name(credentials)

    if l_prod:
        keywords = extracted[1:-1] + c_keywords
    else:
        keywords = extracted[:-1] + c_keywords
    # (athle running) for license FFA

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
    if l_prod:
        cols = df.columns[6:]
    else:
        cols = df.columns[5:]
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
                end_date = pd.to_datetime(begin_date) + pd.Timedelta(weeks=104)

                for date in dates:
                    print(dates)
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
    im: numpy array

    Return:
    -------
    resulting image
    """

    height, width = im.shape

    if height < width:
        new_im = np.rot90(im, k=-1)
    else:
        new_im = im

    return new_im

def create_ngram(txt, n=2, l_text=True):
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
    if l_text:
        ngrams = list(zip(*[txt.split()[i:] for i in range(n)]))
    else:
        # create ngrams and chain them in a string
        ngrams = " ".join([txt[i:i+n] for i in range(len(txt)-n+1)])

    return ngrams

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
        + r"(?:l\d{1}|\d{1}l|ll|\d{2}l\d{1}|\d{2,4})" \
        + r"(?:\d{1,2}|\d{1,2}|\d{2,4})"
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
        theta: approximated width/height ratio of words, filter function is distorted by this factor.Experimentally was found thatsigma is a function of the height of the words (which is related to the height of the line).
        minArea: ignore word candidates smaller than specified area.

    Returns:
        List of tuples. Each tuple contains the bounding box and the image of the segmented word.
    """
    # apply filter kernel
    kernel = createKernel(kernelSize, sigma, theta)
    imgFiltered = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE).astype(np.uint8)
    (_, imgThres) = cv2.threshold(imgFiltered, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    imgThres = 255 - imgThres
    dilationKernel = tuple((np.ones(2)*sigma).astype(int))
    # increase the dilation
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilationKernel)
    imgThres  = cv2.morphologyEx(imgThres, cv2.MORPH_CLOSE, se, iterations=1)
    #imgThres = cv2.dilate(imgThres, dilationKernel, iterations = 1)

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

def len_iter(items):
    """
    item counting
    """
    return sum(1 for _ in items)

def consecutive_key(data, key):
    """
    consecutive key histogram
    """
    hist_list = defaultdict(list)
    count = 0

    for val, run in groupby(data):
        n_consec = len_iter(run)
        hist_list[val].append([count, n_consec])
        count += n_consec

    return hist_list[key]


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """

    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)

    Version history
    ---------------
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True

    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

def horizontal_clustering(components, bin_limit=40):
    """
    group components by line

    Parameters:
    -----------
    components: arrays
        arrays of (x,y,w,h)

    Returns:
    --------
    groups: defaultdict
        grouped dict of arrays of (x,y,w,h)
    """

    x, y, w, h = zip(*components)
    num_bins = np.min([np.ceil((np.max(y) - np.min(y)) / np.median(h)),
                      bin_limit])

    if num_bins == 1:
        num_bins += 1

    bins = np.linspace(np.min(y), np.max(y), num=num_bins, endpoint=True)
    # group by mean of the height
    cut_vals = pd.cut(
        y,
        #np.ceil(np.array(y) + np.array(h)/2.),
        bins=bins,
        include_lowest=True,
        labels = np.arange(num_bins - 1).astype(int))

    groups = defaultdict(list)

    for i, group in enumerate(cut_vals):
        groups[group].append(i)

    return groups

def hough_line_transform(img, minLineLength=100, maxLineGap=80,
                         p_hough=False, linewidth=4):
    """
    Hough line transformation to get rid of lines. Only horizontal lines are
    taken out.
    """

    edges = cv2.Canny(img, 50, 150, apertureSize = 3)
    #edges = canny(img, sigma=1.0)

    if p_hough:
        #lines = probabilistic_hough_line(edges, threshold=100,
        #                                line_length=minLineLength,
        #                                line_gap=maxLineGap)
        lines = cv2.HoughLinesP( edges, 1, np.pi/180, 100,
                                    minLineLength,
                                    maxLineGap )
    else:

        lines = cv2.HoughLines(edges, 1, np.pi/2, 250)
        #h, theta, d = hough_line(edges)#, theta=np.pi/45)
    # image – 8-bit, lines, rho, theta, threshold, minLineLength, maxLineGap
    # draw mask
    if not p_hough:
        # take the width
        n_arb_reconstruct = img.shape[1]
    img_copy = img.copy()
    if lines is not None:
        if p_hough:

            for line in lines:
                 x1,y1,x2,y2 = line[0]
                 #p0, p1 = line
                 #x1 = int(p0[0])
                 #y1 = int(p0[1])
                 #x2 = int(p1[0])
                 #y2 = int(p1[1])

                 # only eliminate horizontal lines
                 diffx = x2 - x1
                 diffy = y2 - y1

                 if not diffx==0:
                     if np.abs(diffy/diffx) < 1:

                         #rr, cc = draw.line(y1, x1, y2, x2)
                         #img_copy[rr, cc] =  255
                         cv2.line(img_copy, (x1,y1), (x2,y2), (255, 255, 255), linewidth)


        else:
            #for _, angle, dist in zip(*hough_line_peaks, h, theta, d, threshold=0.2*h.max())):
            
            for line in lines:
                for rho, theta in line:
                #x1 = 1
                #x2 = img_copy.shape[1] - 1
                #y1 = int((dist - x1 * np.cos(angle)) / np.sin(angle))
                #y2 = int((dist - x2 * np.cos(angle)) / np.sin(angle))
                #_ theta, n_arb_reconstruct = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + n_arb_reconstruct*(-b))
                    y1 = int(y0 + n_arb_reconstruct*(a))
                    x2 = int(x0 - n_arb_reconstruct*(-b))
                    y2 = int(y0 - n_arb_reconstruct*(a))

                    # only eliminate horizontal lines
                    diffx = x2 - x1
                    diffy = y2 - y1

                    if not diffx==0:
                        if np.abs(diffy/diffx) < 1:
                            #y1 = np.clip(y1, 0, img_copy.shape[0] - 1)
                            #y2 = np.clip(y2, 0, img_copy.shape[0] - 1)
                            #rr, cc = draw.line(y1, x1, y2, x2)
                            #img_copy[rr,cc] = 255
                            cv2.line(img_copy, (x1,y1), (x2,y2), (255, 255, 255), linewidth)
    if False:
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(img, cmap="gray")
        ax[1].imshow(img_copy, cmap="gray")
        plt.show()

    return img_copy
