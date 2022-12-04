import cv2
from intelligent_placer_lib.object import load_objects_dataset, detect_objects
from intelligent_placer_lib.contour import detect_contour
from intelligent_placer_lib.placer_scipy import place_one_object, place_many_objects
import intelligent_placer_lib.utils as utils

DEBUG_OUTPUT = False

# main function used by this module
def check_image(path_to_image_on_local_computer):
    # load dataset prepared by tools from images dir
    objects_dataset = load_objects_dataset()

    # load image from path
    image_to_check = cv2.imread(path_to_image_on_local_computer)

    # detect contour
    con = detect_contour(image_to_check)

    # some failure with contour detection
    if (con == None):
        if DEBUG_OUTPUT:
            return "Contour not found"
        else:
            return False        

    if not con.is_convex():
        if DEBUG_OUTPUT:
            return "Contour is not convex"
        else:
            return False        

    # detect objects
    objects = detect_objects(image_to_check, objects_dataset)

    # remove contour from detected objects
    objects = [obj for obj in objects if utils.contours_intersect_area(obj.hull, con.hull) == 0]

    # check obvious things
    contour_area = cv2.contourArea(con.hull)
    contour_radius = cv2.minEnclosingCircle(con.hull)[1]
    for obj in objects:
        obj_area = cv2.contourArea(obj.hull)
        obj_width, obj_height = cv2.minAreaRect(obj.hull)[1]
        if obj_width < obj_height:
            obj_height, obj_width = obj_width, obj_height
        if obj_area > contour_area or obj_width > contour_radius * 2:
            if DEBUG_OUTPUT:
                return "Area of object is too big for contour"
            else:
                return False      

    for obj in objects:
        obj.centerize()
    #check if each object can be placed
    for obj in objects:
        if not place_one_object(obj.hull, con.hull):
            if DEBUG_OUTPUT:
                return "Cannot place one of the objects inside contour"
            else:
                return False            

    if len(objects) <= 1:
        return True

    # hard part - check all objects
    result = place_many_objects([o.hull for o in objects], con.hull)
    if result:
        return True
    else:
        if DEBUG_OUTPUT:
            return "Cannot place all objects inside contour"
        else:
            return False
