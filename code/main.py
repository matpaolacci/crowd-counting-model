import dataframe_utils as dfUtils
import features_construction as fC
from patch_extraction import generatePatchesDataframe
from paper_nn import CcnnModel
import images_utils 
import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm as CM
import tensorflow as tf
import jplumail_neural_network as jplumail_nn
from math import floor
import os

RESIZE_FACTOR = 0.35                                    # The resize factor that will be applied to the images before creating the patches
PATCH_SIZE = 256                                        # The dimensions of the patches, both width and height
SAVES_PATH = "/path/to/checkpoint"                      # Path where the checkpoints will be saved
N_IMAGES = 5                                            # Images to load from directory
TEST_FACTOR = 0.20                                      # The percentage of the images that will be used to test the model.
BATCH_SIZE = 1
EPOCHS = 4
LEARNING_RATE = 0.001
BUILD_DENS_MAPS = True
ARCHITECTURE = 'ccnn'
PATH_TO_ROOT_OF_DATASET = "/path/to/root/of/dataset"

def main(buildDensMaps: bool, architecture: str):
    if buildDensMaps:
        # Load dataset
        df = dfUtils.loadImagesFromDir(
            pathImgsAndJsons=PATH_TO_ROOT_OF_DATASET,
            nImages=N_IMAGES,
            patchSize=PATCH_SIZE,
            transformFun=images_utils.resizeImageAndScaleHeadPoints,
            args=(RESIZE_FACTOR, PATCH_SIZE)
        )

        # Build the density maps
        fC.buildDensityMaps(df, 3.5, 7)
        df = df[['id', 'img', 'dens_map']]
        
        # Patch extraction
        dfForPatchExtraction = df[['img', 'dens_map']]
        patchesDf = generatePatchesDataframe(dfForPatchExtraction, PATCH_SIZE)
        patchesDf.to_pickle(SAVES_PATH + "patch_df_{}imgs.pkl".format(N_IMAGES))

    else:
        patchesDf = pd.read_pickle(SAVES_PATH + "patch_df_{}imgs.pkl".format(N_IMAGES))[:300]
        print("> Dataframe has {} images".format(len(patchesDf)))
    
    print("> Convertion from pandas dataframe to Tensor in progress...")
    nBatches = len(patchesDf)//BATCH_SIZE
    nBatchesTrain = nBatches - floor(nBatches * TEST_FACTOR) # The # total batches minus the # test batches
    
    print("> Creating xTrain tensor...")
    xTrain: tf.Tensor = tf.convert_to_tensor(
        patchesDf['imagePatch'][:BATCH_SIZE * nBatchesTrain].tolist(), 
        dtype=tf.float64
    ) / tf.constant(255.0, dtype=tf.float64)
    print("> xTrain tensor created")

    yTrain: tf.Tensor = tf.convert_to_tensor(
        patchesDf['densPatch'][:BATCH_SIZE * nBatchesTrain].tolist(), 
        dtype=tf.float64
    )
    print("> yTrain tensor created")
    
    print("> Train set has {} instances".format(nBatchesTrain * BATCH_SIZE))
    print("> Test set has {} instances".format( (nBatches-nBatchesTrain) * BATCH_SIZE) )
    
    if architecture == 'jplumail':
        model = jplumail_nn.buildJplumailCcnn(
            learningRate=0.001,
            inputHeight=PATCH_SIZE,
            inputWidth=PATCH_SIZE,
            preprocessingLayer= tf.identity #nn.getNormalizationLayer(xTrain.numpy())
        )
    else:
        model = CcnnModel(
            learningRate=LEARNING_RATE,
            inputHeight=PATCH_SIZE, 
            inputWidth=PATCH_SIZE, 
            inputDepth=3
        )
    
    print("> Training started")
    model.fit(xTrain, yTrain, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)
    print("> Training ended")
    model.save(os.path.join(SAVES_PATH, "{}_lr{}_epoch{}_bsize{}.keras".format(architecture, LEARNING_RATE, EPOCHS, BATCH_SIZE)))
    print("> The model has been saved")
    
    print("> Prepare the data for the mdoel evaluation...")
    del xTrain, yTrain
    xTest: tf.Tensor = tf.convert_to_tensor(
        patchesDf['imagePatch'][BATCH_SIZE * nBatchesTrain : BATCH_SIZE * nBatches].tolist(), 
        dtype=tf.float64
    )/tf.constant(255.0, dtype=tf.float64)
    yTest: tf.Tensor = tf.convert_to_tensor(
        patchesDf['densPatch'][BATCH_SIZE * nBatchesTrain : BATCH_SIZE * nBatches].tolist()
    ) 

    print("> Evaluating of the model..")
    model.evaluate(xTest, yTest)


main(BUILD_DENS_MAPS, ARCHITECTURE)