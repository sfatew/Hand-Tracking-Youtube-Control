from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, Dense, InputLayer, Activation, GlobalAveragePooling2D, BatchNormalization, ReLU, AveragePooling2D, Layer

"""

MSST (Multi-Scale Spatio-Temporal) Module

Architecture:
- Input is fed into 5 parallel branches
- Each branch begins with a 1×1 convolution to reduce dimensions
- Branches have different convolutional patterns:
  * Branch 1: Simple 1×1 convolution (preserves spatial information)
  * Branch 2: 1×1 followed by separate 3×1 and 1×3 convolutions (small receptive field)
  * Branch 3: 1×1 followed by separate 5×1 and 1×5 convolutions (medium receptive field)
  * Branch 4: 1×1 followed by separate 7×1 and 1×7 convolutions (larger receptive field)
  * Branch 5: 1×1 followed by separate 11×1 and 1×11 convolutions (largest receptive field)
- All branches are concatenated at the end to produce output feature maps

Each convolution is typically followed by batch normalization and ReLU activation. This design efficiently captures multi-scale spatial information
"""

class MSST_Layer(Layer):
    def __init__(self, stride, filter1, filter2, filter3, filter4, filter5, **kwargs):
        super(MSST_Layer, self).__init__(**kwargs)
        self.stride = stride
        self.filters = [filter1, filter2, filter3, filter4, filter5]
        self.concat = Concatenate()
        
        # Tạo các nhánh
        self.branch1 = self._make_branch([(1, 1)], filter1, stride)
        self.branch2 = self._make_branch([(1, 1), (3, 3)], [filter1, filter2], stride)
        self.branch3 = self._make_branch([(1, 1), (5, 1), (1, 5)], [filter3]*3, stride)
        self.branch4 = self._make_branch([(1, 1), (7, 1), (1, 7)], [filter4]*3, stride)
        self.branch5 = self._make_branch([(1, 1), (11, 1), (1, 11)], [filter5]*3, stride)

    def _make_branch(self, kernel_sizes, filters, first_stride):
        layers = []
        if isinstance(filters, int):
            filters = [filters] * len(kernel_sizes)
        for i, (kernel, f) in enumerate(zip(kernel_sizes, filters)):
            stride = first_stride if i == 0 else (1, 1)
            layers.append(Conv2D(f, kernel_size=kernel, strides=stride, padding='same'))
            layers.append(BatchNormalization())
            layers.append(ReLU())
        return layers

    def _apply_branch(self, inputs, branch_layers, training):
        x = inputs
        for layer in branch_layers:
            if isinstance(layer, BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

    def call(self, inputs, training=None):
        b1 = self._apply_branch(inputs, self.branch1, training)
        b2 = self._apply_branch(inputs, self.branch2, training)
        b3 = self._apply_branch(inputs, self.branch3, training)
        b4 = self._apply_branch(inputs, self.branch4, training)
        b5 = self._apply_branch(inputs, self.branch5, training)
        return self.concat([b1, b2, b3, b4, b5])

    def get_config(self):
        config = super(MSST_Layer, self).get_config()
        keys = ['filter1', 'filter2', 'filter3', 'filter4', 'filter5']
        config.update({'stride': self.stride, **{k: v for k, v in zip(keys, self.filters)}})
        return config

def build_model(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape = input_shape))

    model.add(MSST_Layer(stride=(1,1), filter1=44, filter2=60, filter3=60, filter4=60,filter5=60))
    model.add(MSST_Layer(stride=(1,1), filter1=48, filter2=80, filter3=80, filter4=80,filter5=80))
    model.add(MSST_Layer(stride=(1,1), filter1=56, filter2=120, filter3=120, filter4=120,filter5=120))

    model.add(AveragePooling2D(pool_size=(3,3), strides=(1,2)))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(MSST_Layer(stride=(1,1), filter1=160, filter2=160, filter3=160, filter4=160,filter5=160))
    model.add(MSST_Layer(stride=(2,1), filter1=72, filter2=200, filter3=200, filter4=200,filter5=200))

    model.add(AveragePooling2D(pool_size=(3,3), strides=(1,2)))
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(MSST_Layer(stride=(1,1), filter1=240, filter2=240, filter3=240, filter4=240,filter5=240))
    model.add(MSST_Layer(stride=(1,1), filter1=320, filter2=320, filter3=320, filter4=320,filter5=320))

    model.add(GlobalAveragePooling2D())
    model.add(BatchNormalization())
    model.add(ReLU())

    model.add(Dense(18, activation='softmax'))
    return model