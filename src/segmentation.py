import numpy as np
import cv2
import matplotlib.patches as patches
from PIL import Image
from helper_functions import (consecutive_key, horizontal_clustering)
from thresholding import Thresholding
from image_preprocessing import ImagePreprocessing
import pdb

class Segmentation:
    def __init__(self, image, morph_close_kernel = (2,2), Th = 3.5,
                connectivity = 8, erode_kernel = (2,2),
                dilate_kernel = (3,3),
                aTl = 5, aTo = 0.4, p_hough = False):
        self.image = np.array(image)
        self.morph_close_kernel = np.ones(morph_close_kernel,np.uint8)
        self.erode_kernel = np.ones(erode_kernel,np.uint8)
        self.dilate_kernel = np.ones(erode_kernel,np.uint8)
        self.Th = Th
        self.connectivity = connectivity
        self.aTl = aTl
        self.aTo = aTo
        self.p_hough = p_hough
        self.new_components_ = None

    def filter_stats_CC(self, img, stats, l_filter=True):
        """
        Establishing connected components based on Zagoris' parameters

        Parameters:
        -----------
        stats: cv2.CC_STAT object
            arrays of (x,y,w,h)

        l_filter: boolean
            True if performing filtering to establish connected components

        Returns:
        --------
        bboxes: numpy ndarray
            contains arrays of (x,y,w,h)

        """

        bboxes = []
        himg, wimg = img.shape

        for i, col in enumerate(stats):
            x = col[0]  #cv2.CC_STAT_LEFT(column)
            y = col[1]  #cv2.CC_STAT_TOP(column)
            w = col[2]  #cv2.CC_STAT_WIDTH(column)
            h = col[3]  #cv2.CC_STAT_HEIGHT(column)

            fn = len(np.where(img[y:y + h, x:x + w] == 0)[1])
            #print( h[i],w[i], h[i],w[i])
            e = np.min([h, w]) / np.max([h, w])
            d = fn / (h * w)

            # 3 criteria in filtering + 1 parameter to avoid taking oversized
            # connected components
            if l_filter:

                if not (np.logical_or.reduce([
                        h < 5, w < 5, d < 0.05, d > 0.9, e < 0.08, w >= 0.8 * wimg,
                        h >= 0.8 * himg
                ])):
                    bboxes.append([x, y, w, h])
            else:
                # do not take boxes that occupy the height of the page
                if not h >= 0.8 * himg:
                    bboxes.append([x, y, w, h])

        return bboxes

    def arlsa(self, img, components, groups):
        """
        Adaptive Run Line Smoothing Algorithm

        Parameters:
        -----------
        components: numpy ndarray
            arrays of (x,y,w,h)

        groups: defaultdict
            dict of grouped components

        Returns:
        --------
        img_copy: numpy ndarray
            image generated after checking on within components

        img_copy_copy: numpy ndarray
            image generated after checking on between components
        """

        self.Th = 3.5
        img_copy = img.copy()
        a_hmin = 0.0

        # within the component, just perform binary output function
        for i, (x, y, w, h) in enumerate(components):
            hmin = (a_hmin*h).astype(int)

            img_copy[y + hmin:y + h - hmin, x:x + w] = 0

        img_copy_copy = img_copy.copy()
        # between components

        for g, q  in groups.items():

            sub_components = sorted(np.array(components)[q], key = lambda x: x[0])

            len_comp = len(sub_components)

            if len_comp>=2:
                # compare with direct neighbour
                for i in range(len_comp -1):
                    (xi, yi, wi, hi) = sub_components[i]
                    (xj, yj, wj, hj) = sub_components[i+1]

                    if xi<=xj:
                        Tl = self.aTl * np.max([wi, wj])
                        To = self.aTo * np.min([hi, hj])

                        xmin = np.min([xi,xj])
                        xmax = np.max([xi+wi,xj+wj])
                        ymin = np.min([yi,yj])
                        ymax = np.min([yi+hi,yj+hj])
                        # hmin is used to color only 0.9 of total hmax in S
                        hmin = (a_hmin*np.min([hi,hj])).astype(int)
                        #pdb.set_trace()
                        #L = len(np.max(img_copy[ymin:ymax, xmin:xmax], axis=0))
                        #L = np.where(img_copy[ymin:ymax, xmin:xmax]==255)[0].size
                        L = np.min([wi,wj])
                        H = np.max([hi, hj]) / np.min([hi, hj])
                        # metric to determine how much it is overlapped, it's positive if it's overlapped
                        O = np.min([yi + hi, yj + hj]) - np.max([yi, yj])

                        if np.logical_and.reduce([L <= Tl, H <= self.Th, O >= To]):
                            img_copy_copy[ymin+hmin:ymax-hmin,xmin:xmax] = 0

        return img_copy, img_copy_copy

    def word_segmentation(self, img, components, axis=0):
        """
        text block segmentation
        Estimation of the threshold that seprates the intra-word distance and
        inter-word distance between two clusters by minimizing the intra-class
        variance between them as in Otsu approach

        Parameters:
        -----------
        img: numpy ndarray
            image array

        components: numpy ndarray
            components after adaptive RLSA, text lines are established

        axis: int
            0 vertical projection, 1 horizontal projection. Default: 0

        Returns:
        --------
        new_components: numpy ndarray
            arrays of (x,y,w,h) that represent words
        """
        # h, w = components.shape

        new_components = []

        for i, (x, y, w, h) in enumerate(components):
            count = np.argmin(img[y:y + h, x:x + w], axis=axis)

            if np.sum(count)>0:
                # replace values less than 0.2*np.max(count) by 0
                #count[count<np.median(count)] = 0
                consec_zeros = consecutive_key(count, 0)

                if not len(consec_zeros) == 0:
                    start, end = self.otsu_hist(consec_zeros)

                    if start is not None and end is not None:
                        # include offset
                        if axis==0:
                            start += x
                            end += x
                            start = np.append(start, x+w)
                            end = np.insert(end, 0, x)

                        elif axis==1:
                            start += y
                            end += y
                            start = np.append(start, y+h)
                            end = np.insert(end, 0, y)

                        for j in range(len(start)):
                            if axis==0:
                                new_w = start[j] - end[j]
                                new_y = y
                                new_x = end[j]
                                new_h = h

                            elif axis==1:
                                new_w = w
                                new_y = start[j]
                                new_x = x
                                new_h = end[j]
                            new_components.append([new_x, new_y, new_w, new_h])
                    else:
                        # already in a word segment
                        new_components.append([x, y, w, h])

        return new_components


    def otsu_hist(self, hist_list):
        """
        thresholding using otsu method to divide the text line block to word
        blocks

        Parameters:
        -----------
        hist_list: numpy ndarray
            arrays of [begin, count], with begin the index where the consecutive
            key begins and count the number of consecutive keys

        Returns:
        --------
        start: array
            an array of starting index of the consecutive key

        end: array
            an array of ending index of the consecutive key
        """

        begin = [v[0] for v in hist_list]
        count = [v[1] for v in hist_list]
        hist = np.zeros(np.max(count) + 1)

        for v in count:
            hist[v] += 1

        sigma_b = np.zeros(len(hist))
        total = np.sum(hist)
        i_hist = np.arange(len(hist)) * hist

        for k in range(len(hist)):
            w0 = np.sum(hist[:k])
            w1 = total - w0

            if (w0 == 0 or w1 == 0):
                # to avoid division by 0
                continue

            mu0 = np.sum(i_hist[:k]) / w0
            mu1 = np.sum(i_hist[k:]) / w1
            sigma_b[k] = w0 * w1 * (mu0 - mu1)**2

        key = np.argmax(sigma_b)

        if key > 0:
            ind = np.where(count > key)
            start = np.array(begin)[ind]
            end = start + np.array(count)[ind] - 1

        else:
            start = None
            end = None

        return start, end

    def run(self, verbose=False):
        """
        run segmentation routine
        """

        img = self.image.copy()

        imgproc = ImagePreprocessing(img, verbose=verbose, p_hough = self.p_hough)
        imgproc.process()
        # Morphological closing
        #linek = np.zeros((11,11),dtype=np.uint8)
        #linek[5,...]=1
        #img = cv2.erode(img, self.erode_kernel)
        #img = cv2.dilate(img, self.dilate_kernel)
        #se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (5,5))

        #img = cv2.morphologyEx(cv2.bitwise_not(imgproc.image), cv2.MORPH_CLOSE,
        #                       se, iterations=1)

        # Step 3: Median blur
        img = imgproc.image#cv2.medianBlur(cv2.bitwise_not(img), 5)
        ##cv2.medianBlur(cv2.bitwise_not(img), 5)

        # Step 4: Connected components
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img,
         self.connectivity, cv2.CV_32S)
        bboxes = self.filter_stats_CC(img, stats)

        # Step 5: ARLSA
        groups = horizontal_clustering(bboxes)
        img_w, img_b = self.arlsa(img, bboxes, groups)
        n_labels_arlsa, labels_arlsa, stats_arlsa, centroids_arlsa = cv2.connectedComponentsWithStats(cv2.bitwise_not(img_b), self.connectivity, cv2.CV_32S)
        self.bboxes_arlsa = self.filter_stats_CC(img_b, stats_arlsa)#,l_filter=False)
        # Step 6: Text block segmentation
        self.new_components_ = self.word_segmentation(img, self.bboxes_arlsa)
        self.image = img
        self.image_b = img_b
