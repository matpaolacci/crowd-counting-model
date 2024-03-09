"""The model taken from the paper"""

from jplumail_neural_network import mae_count, mse_count
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Conv2D, UpSampling2D, MaxPool2D, Input
from tensorflow.keras.optimizers import Adam

class CcnnModel():

    def __init__(self, inputHeight: int, inputWidth: int, inputDepth: int, learningRate: float):
        self.modelInstance = Sequential()

        # ======= Input Layer =======
        self.modelInstance.add(Input(
                shape=(inputHeight, inputWidth, inputDepth)
            )
        )
        
        # ======= Layer 1 =======
        self.modelInstance.add(Conv2D(
            name="layer_1",
            filters = 32,
            padding = 'same',
            kernel_size = 3, 
            activation = 'relu'
        ))
        self.modelInstance.add(MaxPool2D(
            pool_size = 2,
            strides = 2
        ))

        # ======= Layer 2 =======
        self.modelInstance.add(Conv2D(
            name="layer_2",
            filters = 32,
            padding = 'same',
            kernel_size = 3,
            activation = 'relu'
        ))
        self.modelInstance.add(MaxPool2D(
            pool_size = 2,
            strides = 2
        ))

        # ======= Layer 3 =======
        self.modelInstance.add(Conv2D(
            name="layer_3",
            filters = 64,
            padding = 'same',
            kernel_size = 2,
            activation = 'relu'
        ))

        # ======= Layer 4 =======
        self.modelInstance.add(Conv2D(
            name="layer_4",
            filters = 1000,
            kernel_size = 1,
            activation = 'relu'
        ))

        # ======= Layer 5 =======
        self.modelInstance.add(Conv2D(
            name="layer_5",
            filters = 400,
            kernel_size = 1,
            activation = 'relu'
        ))

        # ======= Layer 6 =======
        self.modelInstance.add(Conv2D(
            name="layer_6",
            filters = 1,
            kernel_size = 1
        ))

        self.modelInstance.add(UpSampling2D(
            size = (4,4), 
            interpolation = 'bilinear'
        ))

        self.compile(learningRate=learningRate)

    def fit(self, x, y, batch_size, epochs, **kwargs):
        return self.modelInstance.fit(x, y, batch_size=batch_size, epochs=epochs, **kwargs)

    def predict(self, x, verbose=0):
        return self.modelInstance.predict(x, verbose=verbose)

    def evaluate(self, xTest, yTest):
        return self.modelInstance.evaluate(xTest, yTest)
    

    def compile(self, learningRate: float):
        self.modelInstance.compile(
        optimizer = Adam(learning_rate=learningRate),
        loss = 'MSE',
        metrics = [mae_count, mse_count]
        )

    def summary(self):
        self.modelInstance.summary()

    def save(self, path: str):
        self.modelInstance.save(path)