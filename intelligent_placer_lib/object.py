# File which contains all work with objects

import numpy as np
import cv2
import json
import os

DEBUG_OBJECTS = False

# object representation class
class object:
    # fields
    path = ""  # common part of path to object
    big_image = None  # cv2 image of object
    hull = None  # convex hull
    image = None  # small image of object

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
            cv2.contourArea(contours[i]) >= 0.003 * img_threshold.shape[0] * img_threshold.shape[1]):
            convex_contours.append(cv2.convexHull(contours[i]))

    # add each of them as a contour
    for contour in convex_contours:
        rezult.append(object.from_contour_and_image(contour, image))

    if (DEBUG_OBJECTS):
        cv2.drawContours(image=image, contours=convex_contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                         lineType=cv2.LINE_AA)
        cv2.imshow("contours", image)
        cv2.waitKey(0)
    return rezult