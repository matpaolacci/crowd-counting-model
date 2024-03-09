import pandas as pd
import os
import numpy as np
from typing import Callable
from tqdm.auto import tqdm
import cv2 as cv
from json import load as js_load

IMAGE_DIR_NAME = "images"
JSONS_DIR_NAME = "jsons"

def printDataFrameMemUsage(pandas_df: pd.DataFrame) -> None: 
    '''
        This function prints the statistics about the memory held by the input dataframe.
    '''
    tot_bytes = 0

    memory_info = pandas_df.memory_usage()
    for i in range(len(memory_info)):
        tot_bytes += memory_info[i]

    for row in pandas_df.itertuples(index=False):
        for col in row:
            if isinstance(col, np.ndarray):
                tot_bytes += row[1].size * row[1].itemsize

    tot_mb = round(tot_bytes/(1024**3), 4)
    print("> The pandas dataframe takes {} GB".format(tot_mb))


def showImage(df: pd.DataFrame, image_i: int) -> None:
    '''
        Show the i-th image of the dataframe provided.
    '''
    win_name = "Image {} - (press 0 to close the window)".format(image_i)
    cv.imshow(win_name, df['img'][image_i])
    cv.waitKey(0)
    cv.destroyWindow(win_name)
    cv.waitKey(1)

def getImageHeadPoints(path: str) -> np.ndarray:
    '''
        Load the image json and return the head points as ndarray
        @param path: the path to the image's json.
    '''
    with open(path) as f:
        json_dict = js_load(f)
        np_points = np.array(json_dict['points'])
        return np.round(np_points).astype(np.uint16)


def loadImagesFromDir(
        pathImgsAndJsons: str, 
        nImages: int, 
        patchSize: int,
        bn: bool = False,
        transformFun: Callable[[np.ndarray, np.ndarray], np.ndarray] = None,
        args = None
    ) -> pd.DataFrame:
    '''
        Reads all the images with related metadata (json) which are into the provided dir and 
            return a pandas dataframe that contains all the images with its metadata.

        @param pathImgsAndJsons: The path where are located the directories: images and jsons
        @param n_images: the number of images to load.
        @transform_func: A tranformation function to apply to each images.
        @bn bool: True if you want to load images in bn.
    '''
    df = pd.DataFrame(columns=["id", "img", "head_points", "human_num"])

    imgPaths = os.path.join(pathImgsAndJsons, IMAGE_DIR_NAME)
    imgsFnames = sorted(os.listdir(imgPaths))[:nImages]

    pbar = tqdm(imgsFnames)
    pbar.set_description("> Loading images")
    for filename in pbar:
        
        # takes the image id
        imgId = filename.split(".")[0]

        # read the image
        img = cv.imread(
            os.path.join(imgPaths, filename), 
            cv.IMREAD_GRAYSCALE if bn else cv.IMREAD_COLOR
        )

        h = img.shape[0]
        w = img.shape[1]
        if h < patchSize or w < patchSize:
            continue
        
        # let's take the head points 
        headPoints = getImageHeadPoints(
            os.path.join(pathImgsAndJsons, JSONS_DIR_NAME, imgId + ".json")
        )

        # apply the transformation to image if it has been provided.
        if transformFun != None:
            img, headPoints = transformFun(img, headPoints, *args)

        humanNum = headPoints.shape[0]

        # at last, put the image into the dataframe
        df.loc[len(df)] = [int(imgId), img, headPoints, humanNum]

    print("> Successfully loaded the images!")
    printDataFrameMemUsage(df)

    return df

def applyTransformationToImages(
        df: pd.DataFrame, 
        transformMap: Callable[[np.ndarray], np.ndarray]
    ) -> pd.DataFrame :
    
    tqdm.pandas()
    return df['img'].progress_map(transformMap)