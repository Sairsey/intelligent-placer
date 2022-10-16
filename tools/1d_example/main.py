import matplotlib.pyplot as plt
import numpy as np

contour = [0, 5]
object = [-10, -9]

def f(x):
    obj = [object[0] + x, object[1] + x]
    dist = min(
        abs(contour[0] - obj[0]),
        abs(contour[0] - obj[1]),
        abs(contour[1] - obj[0]),
        abs(contour[1] - obj[1]))
    if contour[0] <= obj[0] and obj[0] <= contour[1]:
        dist = 0

    if contour[0] <= obj[1] and obj[1] <= contour[1]:
        dist = 0

    res = dist ** 2

    if dist != 0:
        res += abs(object[0] - object[1])
    else:
        res += max(contour[0] - obj[0], obj[1] - contour[1], 0)
    
    return res


t = np.arange(-5., 30., 0.1)
y = [f(x) for x in t]
plt.plot(t, y)
plt.show()