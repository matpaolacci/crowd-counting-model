import numpy as np
from tqdm.auto import tqdm
import pandas as pd
from dataframe_utils import printDataFrameMemUsage


def getStartingPoints(h: int, w: int, patchSize: int) -> np.ndarray:
    pointsOnARow= np.arange(0, w, patchSize)
    pointsOnACol= np.arange(0, h, patchSize)

    patchesStartingPoints = []
    for indexOnCol in pointsOnACol:
        for indexOnRow in pointsOnARow:
            patchesStartingPoints.append( (indexOnCol, indexOnRow) )

    patchesStartingPoints = np.array(patchesStartingPoints).reshape(len(pointsOnACol),len(pointsOnARow), 2)
    return patchesStartingPoints
    
def extract(
        img: np.ndarray, 
        densMap: np.ndarray,
		patchDf: pd.DataFrame,
        patchSize: int
    ):
    '''
      @param img: An image in the form of list
    '''
    imgH, imgW = img.shape[0], img.shape[1]
    points = getStartingPoints(imgH, imgW, patchSize)

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

            patchDf.loc[len(patchDf)] = [
                img[sx,sy], 
                densMap[sx,sy]
            ]
  
def generatePatchesDataframe(imgDf: pd.DataFrame, patchSize) -> pd.DataFrame:
    '''
        @param df: The columns order must be following: [ 'img', 'dens_map' ]

        @return: a pandas dataframe with the columns: [ 'imagePatch', 'densPatch']
    '''

    print("> Started the generation of the patches")
    patchsDf = pd.DataFrame(columns=["imagePatch", "densPatch"])

    with tqdm(total=len(imgDf)) as pbar:
        pbar.set_description("> Patch generation in progress")
        for row in imgDf.itertuples(index=False):
            extract(img=row[0], densMap=row[1], patchDf=patchsDf, patchSize=patchSize)

            pbar.update(1)

    print("> Generated {} patches from {} images".format(len(patchsDf), len(imgDf)))
    printDataFrameMemUsage(patchsDf)
    
    return patchsDf