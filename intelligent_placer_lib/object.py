# File which contains all work with objects

import numpy as np
import cv2
import json
import os
from sklearn.cluster import KMeans

import utils

DEBUG_OBJECTS = False
CLUSTERS_AMOUNT = 3

# object representation class
class object:
    # fields
    path = ""  # common part of path to object
    big_image = None  # cv2 image of object
    hull = None  # convex hull
    image = None  # small image of object
    dominant_colors = []

    # contstructor
    def __init__(self, new_path = None):
        if (new_path == None):
            return
        self.path = new_path
        self.big_image = cv2.imread(self.path + ".jpg")
        with open(self.path + "_convex.json", "r") as f:
            self.hull = json.loads(f.read())
        self.hull = np.array(self.hull)
        x, y, w, h = cv2.boundingRect(self.hull)
        # crop image
        self.image = self.big_image[y: y+h, x: x+w]

        mask = np.zeros_like(cv2.split(self.big_image)[0])
        cv2.drawContours(mask, [self.hull], -1, 255, -1)  # Draw filled contour in mask
        img_copy = self.big_image.copy()
        img_copy[mask==0] = (0, 0, 0)
        self.image = img_copy[y: y + h, x: x + w]

        new_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        new_img = new_img[mask!=0]
        # using k-means to cluster pixels
        kmeans = KMeans(n_clusters=CLUSTERS_AMOUNT)
        kmeans.fit(new_img)
        clusters_weight = [0] * CLUSTERS_AMOUNT
        total_sum = 0
        for label in kmeans.labels_:
            clusters_weight[label] += 1
            total_sum += 1
        clusters_weight = [x / total_sum for x in clusters_weight]
        self.dominant_colors = [(clusters_weight[index], [float(x[0]), float(x[1]), float(x[2])]) for index, x in enumerate(kmeans.cluster_centers_)]
        self.dominant_colors.sort(key = lambda x: -x[0])
        medium = np.median(self.hull, axis=0).astype(np.int)
        self.hull = self.hull - medium

    # find rotated rect
    @classmethod
    def from_contour_and_image(self, contour, image):
        new_object = object()
        new_object.hull = contour
        new_object.path = "None"
        new_object.big_image = image
        x, y, w, h = cv2.boundingRect(new_object.hull)

        mask = np.zeros_like(cv2.split(new_object.big_image)[0])
        cv2.drawContours(mask, [new_object.hull], -1, 255, -1)  # Draw filled contour in mask
        img_copy = new_object.big_image.copy()
        img_copy[mask == 0] = (0, 0, 0)
        new_object.image = img_copy[y: y + h, x: x + w]

        new_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        new_img = new_img[mask != 0]
        # using k-means to cluster pixels
        kmeans = KMeans(n_clusters=CLUSTERS_AMOUNT)
        kmeans.fit(new_img)
        clusters_weight = [0] * CLUSTERS_AMOUNT
        total_sum = 0
        for label in kmeans.labels_:
            clusters_weight[label] += 1
            total_sum += 1
        clusters_weight = [x / total_sum for x in clusters_weight]
        new_object.dominant_colors = [(clusters_weight[index], [float(x[0]), float(x[1]), float(x[2])]) for index, x in
                                enumerate(kmeans.cluster_centers_)]
        new_object.dominant_colors.sort(key=lambda x: -x[0])

        medium = np.median(new_object.hull, axis=0).astype(np.int)
        new_object.hull = new_object.hull - medium
        return new_object

# this function will load dataset
def load_objects_dataset():
    directory = "../images/"
    result = []
    for filename in os.listdir(directory):
        if filename.endswith(".json") and not filename.endswith("_convex.json"):
            result.append(object(directory + filename[:-5]))
    return result

def detect_objects(image, objects_dataset):
    rezult = []

    # lets find contours on our image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gauss = cv2.GaussianBlur(img_gray, (5, 5), 0)

    #threshold image
    res, img_threshold = cv2.threshold(img_gauss, 123, 255, cv2.THRESH_BINARY)

    # find countours on binary thresholded image
    contours, hierarchy = cv2.findContours(image=img_threshold, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # collect convex_hulls of contours, and removing obviously incorrect data
    convex_contours = []
    for i in range(len(contours)):
        if (cv2.contourArea(contours[i]) <= 0.5 * img_threshold.shape[0] * img_threshold.shape[1] and
            cv2.contourArea(contours[i]) >= 0.002 * img_threshold.shape[0] * img_threshold.shape[1]):
            convex_contours.append(cv2.convexHull(contours[i]))

    # add each of them as a contour

    new_contours = []

    for contour in convex_contours:
        obj = object.from_contour_and_image(contour, image)
        metrics = [0] * len(objects_dataset)
        for data_index, data in enumerate(objects_dataset):
            for color_index in range(CLUSTERS_AMOUNT):
                #metrics[data_index] += 1 * \
                #    ((obj.dominant_colors[color_index][1][0] / 255 - data.dominant_colors[color_index][1][0] / 255) ** 2 + \
                #    (obj.dominant_colors[color_index][1][1] / 255- data.dominant_colors[color_index][1][1] / 255) ** 2 + \
                #    (obj.dominant_colors[color_index][1][2] / 255- data.dominant_colors[color_index][1][2] / 255) ** 2)
                obj_hsv_color = [utils.rgb_to_hsv(col[1][0], col[1][1], col[1][2]) for col in obj.dominant_colors]
                data_hsv_color = [utils.rgb_to_hsv(col[1][0], col[1][1], col[1][2]) for col in data.dominant_colors]
                metrics[data_index] += 0.5 ** color_index * \
                    ((obj_hsv_color[color_index][0] - data_hsv_color[color_index][0] ) ** 2 + \
                    (obj_hsv_color[color_index][1] - data_hsv_color[color_index][1]) ** 2 + \
                    (obj_hsv_color[color_index][2] - data_hsv_color[color_index][2]) ** 2)

        index_min = min(range(len(metrics)), key=metrics.__getitem__)
        obj.path = objects_dataset[index_min].path
        if DEBUG_OBJECTS:
            cv2.destroyAllWindows()
            print(objects_dataset[index_min].path, index_min, metrics)
            print(obj.dominant_colors)
            print(objects_dataset[index_min].dominant_colors)
            cv2.imshow("Object Colors", utils.generate_colors_image(obj.dominant_colors))
            cv2.imshow("Predicted Colors", utils.generate_colors_image(objects_dataset[index_min].dominant_colors))
            cv2.imshow("Prediceted Object", objects_dataset[index_min].image)
            cv2.imshow("Object", obj.image)
            cv2.waitKey(0)

        if (index_min != 0):
            rezult.append(obj)
            new_contours.append(contour)

    if (DEBUG_OBJECTS):
        cv2.drawContours(image=image, contours=new_contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
            lineType=cv2.LINE_AA)
        cv2.imshow("contours", image)
        cv2.waitKey(0)
    return rezult