import numpy as np
import cv2
from shapely import geometry, ops

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho


def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def rotate_contour(cnt, angle):
    M = cv2.moments(cnt)
    cx = M['m10'] / M['m00']
    cy = M['m01'] / M['m00']

    cnt_norm = cnt - [cx, cy]

    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)

    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)

    xs, ys = pol2cart(thetas, rhos)

    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    return cnt_rotated

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = M['m10']/M['m00']
    cy = M['m01']/M['m00']

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]

    return cnt_scaled

def contours_intersect_area(contour1,contour2):
    polygon1 = [(el[0][0], el[0][1]) for el in contour1]
    polygon2 = [(el[0][0], el[0][1]) for el in contour2]
    polygon1 = geometry.Polygon(polygon1)
    polygon2 = geometry.Polygon(polygon2)

    intersect = polygon1.intersection(polygon2)
    return max(intersect.area, 0)

# Ray tracing
def point_in_contour(x,y,poly):

    n = len(poly)
    inside = False

    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def contours_distance(contour1,contour2):
    cur_dist = 100000
    polygon = [(el2[0][0], el2[0][1]) for el2 in contour2]
    for el1 in contour1:
        point1 = (el1[0][0], el1[0][1])

        if (point_in_contour(point1[0], point1[1], polygon)):
            return 0

        for el2 in contour2:
            point2 = (el2[0][0], el2[0][1])
            vec = (point1[0] - point2[0], point1[1] - point2[1])
            cur_dist = min(cur_dist, np.sqrt(vec[0] ** 2 + vec[1] ** 2))
    return cur_dist
