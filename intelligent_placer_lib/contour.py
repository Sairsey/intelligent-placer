# File which contains all work with contour
import cv2

DEBUG_CONTOURS = False
BG_INTENCITY = 120
BORDER_INTENCITY = 100
CANNY_THRESHOLD_1 = 100
CANNY_THRESHOLD_2 = 200
CONVEX_CHECK_EPS = 0.05

# countour representation class
class countour:
    #fields
    points = []
    hull = []
    area = -1

    # contstructor
    def __init__(self, points):
        self.points = points
        self.hull = cv2.convexHull(self.points)
        self.area = cv2.contourArea(self.points)

    def is_convex(self):
        self.area = cv2.contourArea(self.points)
        convex_area = cv2.contourArea(self.hull)
        return abs(self.area - convex_area) / max(self.area, convex_area) <= CONVEX_CHECK_EPS

def detect_contour(cv2_image):
    #make canny
    img_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    img_gauss = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_canny = cv2.Canny(img_gauss, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2)

    # find countours on binary thresholded canny
    contours, hierarchy = cv2.findContours(image=img_canny, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    image_copy = cv2_image.copy()

    # get amount of nested childs
    childs_number = [[0] * 2 for t in range(len(contours))]
    for i in range(len(contours)):
        start = i
        childs_number[i][0] = i
        while hierarchy[0][start][2] != -1:
            childs_number[i][1] += 1
            start = hierarchy[0][start][2]

    childs_number.sort(key=lambda x : -x[1])

    good_contours = []

    # get all contours with amount of childs are greater than 2
    for i in range(len(childs_number)):
        if childs_number[i][1] >= 2:
            good_contours.append(childs_number[i][0])

    chosen_contour = -1

    # final decision is based on inner color and border color
    for contour_index in good_contours:
        saved_contour_index = contour_index

        #calculate inner point
        M = cv2.moments(contours[contour_index])
        inner_point = [int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])]

        #detect point on border of thiscontour
        cnt = 1
        border_point = contours[contour_index][0][0]
        while hierarchy[0][contour_index][2] != -1:
            contour_index = hierarchy[0][contour_index][2]
            border_point[0] += contours[contour_index][0][0][0]
            border_point[1] += contours[contour_index][0][0][1]
            cnt += 1
        border_point[0] /= cnt
        border_point[1] /= cnt

        # check border color
        border_color = tuple([float(t) for t in cv2_image[border_point[1]][border_point[0]]])
        #image_copy = cv2.circle(image_copy, (point[0], point[1]), 10, color, 4)

        if (DEBUG_CONTOURS):
            print("border", border_color)
        # if border point is too bright
        if border_color[0] >= BORDER_INTENCITY or border_color[1] >= BORDER_INTENCITY or border_color[2] >= BORDER_INTENCITY:
            continue

        inner_color = tuple([float(t) for t in cv2_image[inner_point[1]][inner_point[0]]])
        #image_copy = cv2.circle(image_copy, (inner_point[0], inner_point[1]), 10, inner_color, 4)
        #image_copy = cv2.circle(image_copy, (inner_point[0], inner_point[1]), 11, (255,0,0), 1)
        if (DEBUG_CONTOURS):
            print("inner", inner_color)
        # if background point is too dark
        if inner_color[0] <= BG_INTENCITY or inner_color[1] <= BG_INTENCITY  or inner_color[2] <= BG_INTENCITY:
            continue

        chosen_contour = saved_contour_index

    # if contour cannot be found
    if (chosen_contour == -1):
        return None

    result = countour(contours[chosen_contour])

    if (DEBUG_CONTOURS):
        cv2.drawContours(image=image_copy, contours=[result.hull], contourIdx=-1, color=(0, 255, 0), thickness=2,
                     lineType=cv2.LINE_AA)
        cv2.fillPoly(image_copy, pts=[result.points], color=(255, 255, 255))
        cv2.imshow('Contour debug', image_copy)
        cv2.waitKey(0)

    return result