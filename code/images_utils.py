import numpy as np
from cv2 import resize as cv_resize
from cv2 import INTER_AREA as cv_INTER_AREA

def resizeHeadPoints(headPoints, oldHeight, oldWidth, newHeight, newWidth):
    # getting the head-points array
    n_points = len(headPoints)

    if(n_points <= 0):
        return np.array([])

    # Each array containts the resize constant that we are going to use
    #   to normalize the coordinates x and y of each head point.
    X_c = np.ones(n_points, dtype=np.float32) * ( newWidth - 1) / (oldWidth - 1)
    Y_c = np.ones(n_points, dtype=np.float32) * ( newHeight - 1) / (oldHeight - 1)
    X, Y = np.asarray(headPoints).T
    X, Y = np.around(X * X_c), np.around(Y * Y_c)
    X[X >= newWidth] = newWidth - 1
    Y[Y >= newHeight] = newHeight - 1
    norm_head_points = np.stack((X,Y), axis=1)
    
    return np.ndarray.astype(norm_head_points, dtype = np.uint16)

def resizeImageAndScaleHeadPoints(img, headPoints, resizeFactor, limitMin):
    oldHeight, oldWidth = img.shape[0], img.shape[1]
    newHeight = round(img.shape[0] * resizeFactor)
    newWidth = round(img.shape[1] * resizeFactor)

    # In case the resizing produces an image smaller than limitMin
    if (newHeight <= limitMin) or (newWidth <= limitMin):
        return img, headPoints

    imgResized = cv_resize(img, dsize=(newWidth, newHeight), interpolation=cv_INTER_AREA)
    scaledHeadPoint = resizeHeadPoints(
                            headPoints, 
                            oldHeight=oldHeight, 
                            oldWidth=oldWidth, 
                            newWidth=newWidth,
                            newHeight=newHeight
                        )

    return imgResized, scaledHeadPoint