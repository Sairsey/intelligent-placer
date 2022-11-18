import utils
import cv2
import numpy as np
from scipy.optimize import minimize
from shapely import geometry, ops

DEBUG_PLACER = False

def one_object_optimize_function(data, object, contour):
    posx, posy, rot = data

    #move object by parameter
    object_tmp = utils.rotate_contour(object, rot)
    object_tmp = object_tmp + [posx, posy]

    # First - we have distance between object and contour
    res = utils.contours_distance(object_tmp, contour) ** 2
    res = res

    # Second - we have area of object minus area of intersection between object and contour
    polygon1 = [(el[0][0], el[0][1]) for el in object_tmp]
    polygon2 = [(el[0][0], el[0][1]) for el in contour]
    polygon1 = geometry.Polygon(polygon1)
    polygon2 = geometry.Polygon(polygon2)

    intersect = polygon1.intersection(polygon2)
    res += max((polygon1 - polygon2).area, 0)
    return max(res, 0)

def many_object_optimize_function(data, objects, contour):
    res = 0
    # each object must be inside
    for i in range(len(objects)):
        res += one_object_optimize_function((data[3 * i + 0], data[3 * i + 1], data[3 * i + 2]), objects[i], contour)
    # and they must not overlap
    for i in range(len(objects)):
        posx1, posy1, rot1 = data[3 * i + 0], data[3 * i + 1], data[3 * i + 2]
        first_object = utils.rotate_contour(objects[i], rot1)
        first_object += [posx1, posy1]
        polygon1 = [(el[0][0], el[0][1]) for el in first_object]
        polygon1 = geometry.Polygon(polygon1)
        for j in range(i + 1, len(objects)):
            posx2, posy2, rot2 = data[3 * j + 0], data[3 * j + 1], data[3 * j + 2]
            second_object = utils.rotate_contour(objects[j], rot2)
            second_object += [posx2, posy2]
            polygon2 = [(el[0][0], el[0][1]) for el in second_object]
            polygon2 = geometry.Polygon(polygon2)

            res += polygon1.intersection(polygon2).area

    return res

def place_one_object(obj, con):
    # try to minimize
    res = minimize(one_object_optimize_function, (0.1, 0.1, 0.1) , args=(obj, con), options={'gtol': 1e-7})
    ans = False
    if res.fun <= 1: # because if we intersect with only 1 pixel, it is not fatal
        ans = True
    res = res.x
    if (DEBUG_PLACER):
        print(ans, res)
        new_contour = utils.rotate_contour(obj, res[2]).astype(np.int32)
        new_contour += [int(res[0]), int(res[1])]

        # Create image filled with zeros the same size of original image
        x1, y1, w1, h1 = cv2.boundingRect(con)
        x2, y2, w2, h2 = cv2.boundingRect(new_contour)
        w1 += x1
        h1 += y1
        w2 += x2
        h2 += y2
        blank = np.zeros((max(h1, h2), max(w1, w2), 3))

        blank = cv2.fillPoly(blank, [con], (255, 0, 0))
        blank = cv2.fillPoly(blank, [new_contour], (0, 255, 0))
        cv2.imshow("PLACEMENT", blank)
        cv2.waitKey(0)
    return ans

def place_many_objects(objs, con):
    initial_guess = []
    for o in objs:
        initial_guess.append(0.1) # position
        initial_guess.append(0.1) # position
        initial_guess.append(0.1) # rotation

    # try to minimize
    res = minimize(many_object_optimize_function, initial_guess, args=(objs, con), options={'gtol': 1e-7})
    ans = False
    if res.fun <= 1: # because if we intersect with only 1 pixel, it is not fatal
        ans = True
    res = res.x
    if (DEBUG_PLACER):
        print(ans, res)
        x1, y1, w1, h1 = cv2.boundingRect(con)
        w1 += x1
        h1 += y1
        
        for i in range(len(objs)):
            obj = objs[i]
            new_contour = utils.rotate_contour(obj, res[3 * i + 2]).astype(np.int32)
            new_contour += [int(res[3 * i + 0]), int(res[3 * i + 1])]

            # Create image filled with zeros the same size of original image
            x2, y2, w2, h2 = cv2.boundingRect(new_contour)
            w2 += x2
            h2 += y2
            
            h1 = max(h1, h2)
            w1 = max(w1, w2)
            
        blank = np.zeros((h1, w1, 3))
        blank = cv2.fillPoly(blank, [con], (255, 0, 0))
        colors = [
            [0, 0, 0.9],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 0.5, 0],
            [0, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.1, 0.5, 0.9]]
        for i in range(len(objs)):
            obj = objs[i]
            new_contour = utils.rotate_contour(obj, res[3 * i + 2]).astype(np.int32)
            new_contour += [int(res[3 * i + 0]), int(res[3 * i + 1])]
            blank = cv2.fillPoly(blank, [new_contour], colors[i])
        cv2.imshow("PLACEMENT", blank)
        cv2.waitKey(0)
    return ans
