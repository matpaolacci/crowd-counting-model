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
import pickle

def saveModelHistory(path, history):
    with open(path, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def main(buildDensMaps: bool, 
        architecture: str,
        nImages: int,
        patchSize: int,
        resizeFactor: float,
        learningRate: float,
        epochs: int,
        batchSize: int,
        testSplit: float,
        validationSplit: float,
        checkpointFolder: str,
        datasetFolder: str,
        threadFactor: int = 3,
        nCpu: int = 2,
        maxImagesForTrainAndTest: int = None,
        precision: tf.DType = tf.float32
    ):
    '''
    @param maxImages: the maximum number of images to load from dataframe
    @param nImages: image to load
    @param testFactor: a float that represents the percentage of nImages to use for the tests.
    @param buildDensMaps: Set this to true if you want build the dens maps and store
        the dataframe at checkpointFolder
    @param checkpointFolder: it must not end with '/'
    @param architecture: 'jplumail' or 'ccnn'    
    '''
    if buildDensMaps:
        # Load dataset
        df = dfUtils.loadImagesFromDir(
            pathImgsAndJsons=datasetFolder,
            nImages=nImages,
            patchSize=patchSize,
            transformFun=images_utils.resizeImageAndScaleHeadPoints,
            args=(resizeFactor, patchSize)
        )

        # Build the density maps
        fC.buildDensityMaps(df, sigma=3.5, nCpu=nCpu, threadFactor=threadFactor)
        df = df[['id', 'img', 'dens_map']]
        
        # Patch extraction
        dfForPatchExtraction = df[['img', 'dens_map']]
        patchesDf = generatePatchesDataframe(dfForPatchExtraction, patchSize).sample(frac=1)
        patchesDf.to_pickle(checkpointFolder + "/patch_df_{}imgs.pkl".format(nImages))

    else:
        patchesDf = pd.read_pickle(checkpointFolder + "/patch_df_{}imgs.pkl".format(nImages))[:maxImagesForTrainAndTest]
        print("> Dataframe has {} images".format(len(patchesDf)))
        dfUtils.printDataFrameMemUsage(patchesDf)
    
    nBatches = len(patchesDf)//batchSize
    nBatchesTrain = nBatches - floor(nBatches * testSplit) # The # total batches minus the # test batches
    print("> Creating xTrain tensor...")
    xTrain: tf.Tensor = tf.convert_to_tensor(
        patchesDf['imagePatch'][:batchSize * nBatchesTrain].tolist(), 
        dtype=tf.float32
    ) / tf.constant(255.0, dtype=tf.float32)
    print("> xTrain tensor created")

    print("> Creating yTrain tensor...")
    yTrain: tf.Tensor = tf.convert_to_tensor(
        patchesDf['densPatch'][:batchSize * nBatchesTrain].tolist(), 
        dtype=tf.float32
    )
    print("> yTrain tensor created")
    del patchesDf

    print("> Train set has {} instances".format(nBatchesTrain * batchSize))
    
    if architecture == 'jplumail':
        model = jplumail_nn.buildJplumailCcnn(
            learningRate=0.001,
            inputHeight=patchSize,
            inputWidth=patchSize,
            preprocessingLayer=tf.identity #nn.getNormalizationLayer(xTrain.numpy())
        )
    else:
        model = CcnnModel(
            learningRate=learningRate,
            inputHeight=patchSize, 
            inputWidth=patchSize, 
            inputDepth=3
        )
    
    print("> Training started")
    history = model.fit(xTrain, yTrain, batch_size=batchSize, epochs=epochs, validation_split=validationSplit)
    del xTrain, yTrain
    print("> Training ended")

    model.save(checkpointFolder + "/models/{}_lr{}_epoch{}_bsize{}.keras".format(architecture, learningRate, epochs, batchSize))
    saveModelHistory(checkpointFolder + "/histories/{}_lr{}_epoch{}_bsize{}.history".format(architecture, learningRate, epochs, batchSize), history)
    print("> The model and model's history have been saved")
    
    print("> Prepare the data for the model evaluation...")
    patchesDf = pd.read_pickle(checkpointFolder + "/patch_df_{}imgs.pkl".format(nImages))[:maxImagesForTrainAndTest]
    print("> Test set has {} instances".format( (nBatches-nBatchesTrain) * batchSize) )

    xTest: tf.Tensor = tf.convert_to_tensor(
        patchesDf['imagePatch'][batchSize * nBatchesTrain : batchSize * nBatches].tolist(), 
        dtype=tf.float32
    )/tf.constant(255.0, dtype=tf.float32)
    yTest: tf.Tensor = tf.convert_to_tensor(
        patchesDf['densPatch'][batchSize * nBatchesTrain : batchSize * nBatches].tolist()
    ) 
    
    del patchesDf
    
    print("> Evaluating of the model")
    model.evaluate(xTest, yTest)
    del xTest, yTest
