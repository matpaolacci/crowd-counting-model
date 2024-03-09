import matplotlib.pyplot as plt
import numpy as np
import pickle

def plotHistory(historyPath: str):
    with open(historyPath, "rb") as file_pi:
        history = pickle.load(file_pi)
    

    # list all data in history
    print(history.keys())

    loss = np.array(history['loss'])
    history['loss'] = loss[loss<1].tolist()
    # summarize history for MSE
    plt.plot(history['mse_count'])
    plt.plot(history['val_mse_count'])
    plt.title('model MSE Count')
    plt.ylabel('mse_count')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # summarize history for MAE
    plt.plot(history['mae_count'])
    plt.plot(history['val_mae_count'])
    plt.title('model MAE Count')
    plt.ylabel('mae_count')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def printMaeMse(historyPath: str):
    with open(historyPath, 'rb') as f:
        history = pickle.load(f)
    
    print("MSE: {}".format(history['mse_count'][-1]))
    print("MAE: {}".format(history['mae_count'][-1]))
    print("VAL_MSE: {}".format(history['val_mse_count'][-1]))    
    print("VAL_MAE: {}".format(history['val_mae_count'][-1]))
