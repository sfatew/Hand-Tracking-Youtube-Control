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

    def build(self, input_shape):
        self.branch1 = self._make_branch([(1, 1)], self.filters[0], self.stride)
        self.branch2 = self._make_branch([(1, 1), (3, 3)], [self.filters[0], self.filters[1]], self.stride)
        self.branch3 = self._make_branch([(1, 1), (5, 1), (1, 5)], [self.filters[2]]*3, self.stride)
        self.branch4 = self._make_branch([(1, 1), (7, 1), (1, 7)], [self.filters[3]]*3, self.stride)
        self.branch5 = self._make_branch([(1, 1), (11, 1), (1, 11)], [self.filters[4]]*3, self.stride)
        super().build(input_shape)

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
