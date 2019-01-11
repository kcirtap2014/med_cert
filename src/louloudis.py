import cv2
import numpy as np
import matplotlib.pylab as plt
import matplotlib.patches as patches
import pdb
from collections import defaultdict
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

class Textline:

    def __init__(self, rho1, theta, rho2, n_arb_reconstruct):
        a = np.cos(theta)
        b = np.sin(theta)

        x10 = a*rho1
        y10 = b*rho1
        x11 = x10 + n_arb_reconstruct*(-b)
        x12 = x10 - n_arb_reconstruct*(-b)
        y11 = y10 + n_arb_reconstruct*(a)
        y12 = y10 - n_arb_reconstruct*(a)

        x20 = a*rho2
        y20 = b*rho2

        x21 = x20 + n_arb_reconstruct*(-b)
        x22 = x20 - n_arb_reconstruct*(-b)
        y21 = y20 + n_arb_reconstruct*(a)
        y22 = y20 - n_arb_reconstruct*(a)

        a1 = (y12 - y11)/(x12 - x11)
        a2 = (y22 - y21)/(x22 - x21)

        self.theta = theta
        self.rho1 = rho1
        self.rho2 = rho2

        self.line1 = (lambda x: a1*x+y11)
        self.line2 = (lambda x: a2*x+y21)


class Louloudis:

    def __init__(self, img, connectivity = 8):
        self.img = img
        self.connectivity = connectivity
        self.plot = True
        self.subcomponents = pd.DataFrame(columns = ['x','y','w','h','x_cent','y_cent','subset', 'text_line'])


    def preprocess(self):

        self.subset1_img = 255 - np.zeros((self.img.shape[0],self.img.shape[1]), np.uint8)
        self.subset2_img = 255 - np.zeros((self.img.shape[0],self.img.shape[1]), np.uint8)
        self.subset3_img = 255 - np.zeros((self.img.shape[0],self.img.shape[1]), np.uint8)
        neg_img = 255 - self.img
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(neg_img,
         self.connectivity, cv2.CV_32S)

        # eliminate too big or too small bounding boxes
        mod_stats = [(i, [x,y,w,h,a]) for i, (x,y,w,h,a) in enumerate(stats) if
                                    np.logical_and.reduce([h < 0.9*self.img.shape[0],
                                                           w < 0.9*self.img.shape[1],
                                                           h > 2, w > 2])]

        self.index_stats, arr_stats = list(zip(*mod_stats))
        x, y, w, h, a = list(zip(*arr_stats))
        self.ah = int(np.mean(h))
        self.aw = int(np.mean(w))

        #fig, ax = plt.subplots()
        #ax.imshow(self.img, cmap="gray")

        for i, arr in mod_stats:
            (x, y, w, h, a) = tuple(arr)

            if (h >= 0.5 * self.ah and h < 3 * self.ah) and (w >= 0.5 * self.aw):
                x_temp = x
                self.subcomponents.loc[i] = [x,y,w,h,centroids[i][0], centroids[i][1],1,[]]

                # partitioning
                while x_temp < x + w:

                    # self.subset1.append([x_temp, y, self.aw, h])
                    #rect = patches.Rectangle((x_temp,y),aw,h,linewidth=1, edgecolor='b',
                    #                     facecolor='none')
                    #ax.add_patch(rect)
                    cv2.rectangle(self.subset1_img, (x_temp, y), (x_temp+self.aw, y+h), (0,0,0), 4)
                    x_temp += self.aw


            elif (h >= 3 * self.ah):
                self.subcomponents.loc[i] = [x,y,w,h,centroids[i][0], centroids[i][1],2,[]]
                #rect = patches.Rectangle((x,y),w,h,linewidth=1, edgecolor='g',
                #                     facecolor='none')
                #ax.add_patch(rect)
                cv2.rectangle(self.subset2_img, (x, y), (x+w, y+h), (0,0,0), 4)

            elif (h < 3 * self.ah and w < 0.5 * self.aw) or (h < 0.5 * self.ah and w > 0.5 * self.aw):
                self.subcomponents.loc[i] = [x,y,w,h,centroids[i][0], centroids[i][1],3,[]]
                #rect = patches.Rectangle((x,y),w,h,linewidth=1, edgecolor='r',
                #                     facecolor='none')
                #ax.add_patch(rect)
                cv2.rectangle(self.subset3_img, (x, y), (x+w, y+h), (0,0,0), 4)

    def hough_lines_acc(self, img, rho_resolution=1, theta_resolution=1):
        ''' A function for creating a Hough Accumulator for lines in an image. '''
        height, width = img.shape # we need heigth and width to calculate the diag
        img_diagonal = np.ceil(np.sqrt(height**2 + width**2)) # a**2 + b**2 = c**2
        rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
        thetas = np.deg2rad(np.arange(85, 95, theta_resolution))

        # create the empty Hough Accumulator with dimensions equal to the size of
        # rhos and thetas
        H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
        y_idxs, x_idxs = np.nonzero(img) # find all edge (nonzero) pixel indexes

        for i in range(len(x_idxs)): # cycle through edge points
            x = x_idxs[i]
            y = y_idxs[i]

            for j in range(len(thetas)): # cycle through thetas and calc rho
                rho = int((x * np.cos(thetas[j]) +
                           y * np.sin(thetas[j])) + img_diagonal) // rho_resolution
                H[rho, j] += 1

        return H, rhos, thetas

    def hough_transform(self):
        # building the accumulator array
        #(2) detect straight line segments using the Hough Transform
        edges = cv2.Canny(self.subset1_img, 50, 150, apertureSize = 3)
        p_step_size = int(0.2*self.ah)

        H, rhos, thetas = self.hough_lines_acc(cv2.bitwise_not(self.subset1_img), rho_resolution = p_step_size)
        text_line = defaultdict()

        H_copy = H.copy()
        n1 = int(0.3*(np.max(H)))
        n2 = int(0.4*(np.max(H)))
        #n1 =  np.percentile(H, 50)
        #n2 =  np.percentile(H, 60)
        count = 0
        max_H = np.max(H_copy)
        pixel_sep = 5

        while max_H > n1:
            row, col = np.unravel_index(np.argmax(H_copy), H_copy.shape)
            H_copy[row,col] = 0

            if max_H > n2:
                # you just need both extreme points
                row_n5 = np.clip(row - pixel_sep, 0, len(rhos)-1)
                row_p5 = np.clip(row + pixel_sep, 0, len(rhos)-1)

                text_line[count] = Textline(rhos[row_n5],thetas[col],rhos[row_p5],self.img.shape[1])

                H_copy[row_n5:row_p5,col] = 0
            else:
                #text_line[0] main component
                if np.rad2deg(np.abs(text_line[0].theta - thetas[col]))<2:
                    row_n5 = np.clip(row - pixel_sep, 0, len(rhos)-1)
                    row_p5 = np.clip(row + pixel_sep, 0, len(rhos)-1)
                    text_line[count] = Textline(rhos[row_n5],thetas[col],rhos[row_p5],self.img.shape[1])
                    H_copy[row_n5:row_p5,col] = 0

            max_H = np.max(H_copy)
            count += 1

        if self.plot:
            n_arb_reconstruct = self.img.shape[1]
            test = self.img.copy()#255 - np.zeros((self.img.shape[0],self.img.shape[1]), np.uint8)
            #line1 = text_line[1][0]
            #line2 = text_line[1][-1]
            #new_line = [line1,line2]
            interval_index = []

            count_line = defaultdict(int)
            subset1 = self.subcomponents[self.subcomponents.subset==1]
            h_text_line = 2*pixel_sep*p_step_size

            for index, row in subset1.iterrows():
            #for j, subcomp in self.subset1:
            #for i, (j, subcomp) in enumerate(subset1):

                for k, lines in text_line.items():
                    yval1 = lines.line1(row["x"])
                    yval2 = yval1 + h_text_line
                    y_text_line = np.arange(int(yval1),int(yval2))
                    y_subcomponent = np.arange(row['y'],row['y']+row['h'])

                    inter = len(set(y_text_line).intersection(set(y_subcomponent)))
                    #print(y_text_line, y_subcomponent, inter/row['h'])
                    # check for intersection
                    if inter/row['h']>=0.5:
                        count_line[k] += 1
                        self.subcomponents.loc[index,"text_line"].append(k)

                    #yval2 = lines.line2(x+w)
                    #self.subset1_img[row['y']:row['y']+row['h'],row['x']:row['x']+row['w']]
                    # the subcomponent must be in the line
                    #if self.subset1_img[row['y']:row['y']+row['h'],row['x']:row['x']+row['w']]:

                    #if (row["y"]+0.25*row["h"]>=yval1
                    #        and row["y"]+0.75*row["h"]<=yval2):
                    #    count += 1
                    #    self.subcomponents.loc[index,"text_line"].append(k)
                    #if row["y_cent"]>=yval1 and row["y_cent"]<=yval2:
                    #    count_plot += 1
                    #    self.subcomponents.loc[index,"text_line"].append(k)
                    #elif row["y"]>=yval1 or row["y"]+row["h"]<=yval2:
                    #    count += 1
                    #    self.subcomponents.loc[index,"text_line"].append(k)
            #print(count_plot)

            def most_pop_line(x):
                if not x==[]:
                    inter = [(el, count_line[el]) for el in x]
                    sorted_by_value = sorted(inter, key=lambda kv: kv[1], reverse=True)
                    most_pop = sorted_by_value[0][0]
                else:
                    most_pop = np.NaN

                return most_pop

            subset1.loc[:,"len_text_line"] = subset1["text_line"].apply(lambda x: len(x))
            subset1.loc[:,"first_text_line"] = subset1["text_line"].apply(most_pop_line)
            zero_subcomp = subset1[subset1["len_text_line"]==0]
            keepTextLines = [int(x) for x in subset1["first_text_line"].unique() if not np.isnan(x)]

            #all_rhos = [(x.rho1,x.theta) for k, x in text_line.items() if k in keepTextLines]
            #sorted_all_rhos = sorted(all_rhos, key = lambda x: x[0])
            #pdb.set_trace()
            if True:
                # clustering remaining unassigned
                keep_cols = ["x","y","h","y_cent"]
                X = zero_subcomp[keep_cols]
                #X.plot.scatter(x="x_cent",y="y_cent")
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                scores = []
                offset = 2
                max_classes = np.min([len(keepTextLines), len(X)])

                for n_cluster in np.arange(offset, max_classes):
                    clf = KMeans(n_clusters=n_cluster)
                    clf.fit_transform(X_scaled)
                    score = silhouette_score(X_scaled, clf.labels_)
                    scores.append(score)

                best_n_cluster = np.argmax(scores) + offset
                clf_best = KMeans(n_clusters = best_n_cluster)
                clf_best.fit_transform(X_scaled)
                zero_subcomp.loc[:,"cluster"] = clf_best.labels_
                label_cluster_dict = defaultdict()

                for i in range(len(clf_best.cluster_centers_)):
                    item = scaler.inverse_transform(clf_best.cluster_centers_[i])

                    for k, lines in text_line.items():
                        if k in keepTextLines:
                            # attribute the line to the ones in keepTextLines
                            yval1 = lines.line1(item[keep_cols.index("x")])
                            yval2 = yval1 + h_text_line

                            if item[keep_cols.index("y_cent")]>=yval1 and item[keep_cols.index("y_cent")]<=yval2:
                                label_cluster_dict[i] = k

                # the remaining subcomponents are not associated to any text line
                for unique_val in zero_subcomp.cluster.unique():

                    if unique_val not in list(label_cluster_dict.keys()):
                        x_cluster = zero_subcomp[zero_subcomp.cluster==unique_val]["x"]
                        y_cluster = zero_subcomp[zero_subcomp.cluster==unique_val]["y"]
                        h_cluster = zero_subcomp[zero_subcomp.cluster==unique_val]["h"]
                        mean_y_cluster = np.mean(y_cluster)
                        mean_h_cluster = np.mean(h_cluster)
                        ymax_cluster = mean_y_cluster + mean_h_cluster

                        x = np.min(x_cluster)
                        x_w = np.max(x_cluster)

                        theta_cluster = np.arctan2(abs((x - x_w)), abs((mean_y_cluster - ymax_cluster))) + 90
                        # add 90 because by default that's a horizontal line

                        rho1_cluster = np.sqrt(np.dot(x_cluster,x_cluster) + np.dot(y_cluster,y_cluster))
                        rho2_cluster = np.sqrt(np.dot(x_cluster,x_cluster) + np.dot(y_cluster+h_cluster,y_cluster+h_cluster))
                        text_line[count] = Textline(rho1_cluster,theta_cluster,rho2_cluster,self.img.shape[1])
                        label_cluster_dict[unique_val] = count
                        count += 1

                # creation of new lines
                subset1.loc[zero_subcomp.index,"first_text_line"] = zero_subcomp.cluster.apply(lambda x: label_cluster_dict[x])

            # check unassigned components
            finalTextLines = [int(x) for x in subset1["first_text_line"].unique() if not np.isnan(x)]
            selected_text_lines = [ (k,v) for k, v in text_line.items() if k in finalTextLines ]
            min_ydist = self.img.shape[0]
            subset3 = self.subcomponents[self.subcomponents.subset==3]

            for index, row in subset3.iterrows():
                for k, lines in selected_text_lines:
                    yval1 = lines.line1(row["x"])
                    y_line_cent = yval1 + h_text_line/2
                    ydist = np.abs(y_line_cent - row["y_cent"])

                    if ydist<min_ydist:
                        min_ydist = ydist
                        index_text_line = k

                subset3.loc[index, "first_text_line"] = index_text_line

            if self.plot:
                for index, row in subset1.iterrows():
                    x1 = row["x"]
                    x2 = row["x"]+row["h"]
                    index_text_line = row["first_text_line"]

                    y1 = int(text_line[index_text_line].line1(x1))
                    y2 = int(text_line[index_text_line].line1(x2))
                    y12 = int(text_line[index_text_line].line2(x1))
                    y22 = int(text_line[index_text_line].line2(x2))

                    cv2.line(test, (x1,y1), (x2,y2), (0,0,0), 3)
                    cv2.line(test, (x1,y12), (x2,y22), (0,0,0), 3)

                for index, row in subset3.iterrows():
                    x1 = row["x"]
                    x2 = row["x"]+row["h"]
                    index_text_line = row["first_text_line"]

                    y1 = int(text_line[index_text_line].line1(x1))
                    y2 = int(text_line[index_text_line].line1(x2))
                    y12 = int(text_line[index_text_line].line2(x1))
                    y22 = int(text_line[index_text_line].line2(x2))

                    cv2.line(test, (x1,y1), (x2,y2), (0,0,0), 3)
                    cv2.line(test, (x1,y12), (x2,y22), (0,0,0), 3)

                fig, ax = plt.subplots(1,3)
                ax[0].imshow(test, cmap="gray")
                ax[1].imshow(self.subset1_img, cmap="gray")
                ax[2].imshow(self.subset3_img, cmap="gray")


                plt.show()
                print("Step 3: Hough Transform applied.")
                pdb.set_trace()

    def post_processing(self):
        pass
