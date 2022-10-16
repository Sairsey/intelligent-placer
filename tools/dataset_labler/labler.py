# this code will provide GUI for user for creating polygon for each object
from tkinter import filedialog as fd
import cv2
import numpy as np
import json

# global variable which stores image loaded by opencv
# because this is just a tool, I think it is not a problem
picture_name = ""
image = None
polygon = []

# callback for pressing mouse
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image,(x,y),1,(255,0,0),-1)
        polygon.append([x, y])
# callback for polygon saving
def save_polygon():
    polygon_name = picture_name[:-3] + "json"
    with open(polygon_name, "w") as f:
        f.write(json.dumps(polygon))

if __name__ == "__main__":
    filetypes = (
        ('JPG files', '*.jpg'),
        ('PNG files', '*.png')
    )

    # ask for image
    picture_name = fd.askopenfilename(
        title = "Open a image",
        initialdir="../../images",
        filetypes=filetypes
    )

    #show image via opencv window api

    #load image
    image = cv2.imread(picture_name)

    #create Window and set callback
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_circle)
    #main loop
    while True:
        #if q pressed - exit
        pressed_button = cv2.waitKey(25)

        if pressed_button & 0xFF == ord('q'):
            break
        elif pressed_button & 0xFF == ord('s'):
            save_polygon()

        #prepare frame buffer
        tmpImg = image.copy()
        if (len(polygon) > 1):
            #draw polygon
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(tmpImg, [pts], True, (255, 0, 0), 2)
        #show image
        cv2.imshow('Image', tmpImg)

    cv2.destroyAllWindows()  # destroys the window showing image