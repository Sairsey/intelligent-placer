# this code will go over all jsons in images directory and make convex hulls out of them
import cv2
import json
import os
import numpy as np

if __name__ == "__main__":
    directory = "../../images"
    for filename in os.listdir(directory):
        if filename.endswith(".json") and not filename.endswith("_convex.json"):
            fullname = os.path.join(directory, filename)
            #load data from file
            data = []
            with open(fullname, "r") as f:
                data = json.loads(f.read())
            data = np.array(data)

            #build convexHull
            convex = cv2.convexHull(data, False)

            #show image with both hulls
            image_name = fullname[:-4] + "jpg"

            image = cv2.imread(image_name)

            pts = data
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (255, 0, 0), 2)

            pts = convex
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 0, 255), 2)

            cv2.imshow('Image', image)

            while cv2.waitKey(0) & 0xFF != ord('q'):
                pass

            convexname = fullname[:-5] + "_convex.json"
            #save new element
            with open(convexname, "w") as f:
                f.write(json.dumps(convex.tolist()))
            cv2.destroyAllWindows()  # destroys the window showing image
        else:
            continue


