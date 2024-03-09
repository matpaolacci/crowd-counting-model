import numpy as np
from patch_extraction import getStartingPoints
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import cm as CM
from tensorflow.keras.models import load_model
from jplumail_neural_network import mae_count, mse_count
import json
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

def getJson(path):
    with open(path) as f:
        js = json.load(f)
    return js

def evaluateModelOnPatches(modelPath: str, datasetPath: str):
    dataset = pd.read_pickle(datasetPath)
    dataset = dataset[len(dataset)-2048:] 
    print("> Evaluate model on {} instances".format(len(dataset)))
    model = load_model(modelPath, custom_objects = {"mae_count": mae_count, "mse_count": mse_count})
    xTest: tf.Tensor = tf.convert_to_tensor(
        dataset['imagePatch'].tolist(),
        dtype=tf.float32
    ) / tf.constant(255.0, dtype=tf.float32)
    yTest: tf.Tensor = tf.convert_to_tensor(
        dataset['densPatch'].tolist(),
        dtype=tf.float32
    )
    model.evaluate(xTest, yTest)

def evaluateModelOnImagesRangeWithMAE(modelPath: str, dataFolder: str, idStart: int, idEnd: int, patchSize: int):
    '''
        Evaluate the model on the given range of the IDs of the images with MAE
    '''
    model: tf.keras.Model = load_model(modelPath, custom_objects = {"mae_count": mae_count, "mse_count": mse_count})
    err = []
    numOfHumans = []
    for i in tqdm(range(idStart, idEnd+1)):
        imagePath = dataFolder + "/images/{}.jpg".format(str(i).zfill(4))
        jsonPath = dataFolder + "/jsons/{}.json".format(str(i).zfill(4))
        groundTruthHumanNumber = int(getJson(jsonPath)['human_num'])
        numOfHumans.append(groundTruthHumanNumber) 
        img: np.ndarray = cv.imread(imagePath)/255.0
        w, h = img.shape[1], img.shape[0]
        patches = extract(img, patchSize)
        # the density maps of each patch
        predOfPatches = model.predict(patches, verbose=0).reshape((patches.shape[0], patches.shape[1], patches.shape[2]))
        assembledDensMap = composeDensMap(predOfPatches, patchSize, h, w)
        predictionOfNumberOfHumans = np.abs(assembledDensMap.sum())
        err.append(abs(groundTruthHumanNumber - predictionOfNumberOfHumans))
    
    numOfHumans = np.array(numOfHumans)
    err = np.array(err)
    print("The MAE for the images from {} to {} is {}".format(idStart, idEnd, err.sum()/err.shape[0]))
    print("The mean number of humans over the images range is {}".format(numOfHumans.mean()))
    print("The MAX number of humans over the images range is {}".format(numOfHumans.max()))
    print("The MIN number of humans over the images range is {}".format(numOfHumans.min()))

def predictDensityMap(modelPath: str, dataFolder: str, imageId: str, patchSize: int):
    '''
        @param dataFolder: where are located the folder '/images' and '/jsons/'
    '''
    model = load_model(modelPath, custom_objects = {"mae_count": mae_count, "mse_count": mse_count})
    imagePath = dataFolder + "/images/{}.jpg".format(imageId)
    jsonPath = dataFolder + "/jsons/{}.json".format(imageId)
    jsImg = getJson(jsonPath)
    img: np.ndarray = cv.imread(imagePath)/255.0
    img = img.astype(np.float64)
    cv.imshow("images", img)
    cv.waitKey(1)
    w, h = img.shape[1], img.shape[0]
    patches = extract(img, patchSize)
    densMaps = model.predict(patches).reshape((patches.shape[0], patches.shape[1], patches.shape[2]))
    assembledDensMap = composeDensMap(densMaps, patchSize, h, w)
    nHeadsFromEntire = np.abs(assembledDensMap.sum())
    print("Number of heads from dens map recomposed from patches: {}".format(nHeadsFromEntire))
    print("Ground truth head number: {}".format(jsImg['human_num']))
    del jsImg

    plt.imshow(assembledDensMap, cmap=CM.jet)
    plt.show()
    cv.destroyAllWindows()
    cv.waitKey(1)

def extract(img: np.ndarray, patchSize: int) -> np.ndarray:
    imgH, imgW = img.shape[0], img.shape[1]
    points = getStartingPoints(imgH, imgW, patchSize)
    patches = []

    for row in range(len(points)):
        for col in range(len(points[row])):
            x,y = points[row][col].tolist()

            if row == len(points)-1:
                sx=slice(img.shape[0]-patchSize,img.shape[0],None)
            else:
                sx=slice(x,x+patchSize,None)
            
            if col == len(points[row])-1:
                sy=slice(img.shape[1]-patchSize,img.shape[1],None)
            else:
                sy=slice(y,y+patchSize,None)     
            patches.append(img[sx,sy])

    return np.array(patches)


def composeDensMap(densPatches: np.ndarray, patchSize: int, imageH: int, imageW: int):
    points = getStartingPoints(imageH, imageW, patchSize)

    densMap = np.zeros((imageH, imageW), dtype=np.float64)
    i = 0
    for row in range(len(points)):
        for col in range(len(points[row])):
            x,y = points[row][col].tolist()

            if row == len(points)-1:
                sx=slice(x,imageH,None)
            else:
                sx=slice(x,x+patchSize,None)
            
            if col == len(points[row])-1:
                sy=slice(y,imageW,None)
            else:
                sy=slice(y,y+patchSize,None)     

            startY = slice(patchSize - densMap[sx, sy].shape[0], patchSize, None)
            startX = slice(patchSize - densMap[sx, sy].shape[1], patchSize, None)
            densMap[sx, sy] = densPatches[i][startY, startX]
            i += 1

    return densMap
