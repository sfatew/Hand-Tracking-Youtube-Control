import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

ACTIONS = [
    "Doing other things", "No gesture", 'Rolling Hand Backward', 'Rolling Hand Forward', 'Shaking Hand',
    'Sliding Two Fingers Down', 'Sliding Two Fingers Left', 'Sliding Two Fingers Right', 'Sliding Two Fingers Up',
    'Stop Sign', 'Swiping Down','Swiping Left', 'Swiping Right', 'Swiping Up',
    'Thumb Down', 'Thumb Up',
    'Turning Hand Clockwise', 'Turning Hand Counterclockwise'
]
NUM_CLASSES = len(ACTIONS)

# Model / Data shape parameters
MAX_FRAMES = 37
NUM_KEYPOINTS = 48
NUM_COORDINATES = 3

# Training parameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

class NonLocalSpatialShift(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        B, T, N, C_in = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        
        num_channels_static = inputs.shape[-1]
        if num_channels_static is None: 
            num_channels_static = C_in 

        shifted_features_list = []
        for c_idx in range(num_channels_static):
            shift_val = tf.cast(c_idx % N, dtype=tf.int32)
            rolled_channel = tf.roll(inputs[:, :, :, c_idx], shift=shift_val, axis=2)
            shifted_features_list.append(tf.expand_dims(rolled_channel, axis=-1))

        shifted_inputs = tf.concat(shifted_features_list, axis=-1)
        return shifted_inputs

class SpatialShiftConvBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.spatial_shift = NonLocalSpatialShift()
        self.pw_conv = layers.Conv2D(filters, kernel_size=(1, 1), padding='same', data_format='channels_last')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs, training=False):
        x = self.spatial_shift(inputs)
        x = self.pw_conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x
    
class NaiveTemporalShift(layers.Layer):
    def __init__(self, u=1, **kwargs):
        super().__init__(**kwargs)
        self.u = u
        self.num_partitions = 2 * u + 1

    def call(self, inputs):
        B, T, N, C_in = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        
        num_channels_static = inputs.shape[-1]
        if num_channels_static is None:
             num_channels_static = C_in

        base_split_size = num_channels_static // self.num_partitions
        remainder = num_channels_static % self.num_partitions
        size_splits = [base_split_size + 1] * remainder + [base_split_size] * (self.num_partitions - remainder)
        
        size_splits = [s for s in size_splits if s > 0]
        actual_num_partitions = len(size_splits) 

        if actual_num_partitions == 0 : 
            return inputs

        channel_partitions = tf.split(inputs, num_or_size_splits=size_splits, axis=-1)
        
        shifted_partitions = []
        if actual_num_partitions == 1:
             return inputs
        
        for i in range(actual_num_partitions):
            if actual_num_partitions % 2 == 1: 
                current_u = (actual_num_partitions -1) // 2
                shift_val = i - current_u
            else: 
                shift_val = i - self.u
                if i >= actual_num_partitions // 2 and self.u > (actual_num_partitions // 2 -1) : 
                    shift_val = i - (actual_num_partitions // 2 -1)


            partition = channel_partitions[i]
            rolled_partition = partition
            
            if shift_val > 0:
                core = partition[:, :-shift_val, :, :]
                padding = tf.zeros_like(partition[:, :shift_val, :, :])
                rolled_partition = tf.concat([padding, core], axis=1)
            elif shift_val < 0:
                s_abs = -shift_val
                core = partition[:, s_abs:, :, :]
                padding = tf.zeros_like(partition[:, :s_abs, :, :])
                rolled_partition = tf.concat([core, padding], axis=1)
            
            shifted_partitions.append(rolled_partition)

        if not shifted_partitions:
            return inputs
            
        shifted_inputs = tf.concat(shifted_partitions, axis=-1)
        return shifted_inputs


class TemporalShiftConvBlock(layers.Layer):
    def __init__(self, filters, u=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.u = u
        self.temporal_shift = NaiveTemporalShift(u=self.u)
        self.pw_conv = layers.Conv2D(filters, kernel_size=(1,1), padding='same', data_format='channels_last')
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

    def call(self, inputs, training=False):
        x = self.temporal_shift(inputs)
        x = self.pw_conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x
    
class ShiftGCNBlock(layers.Layer):
    def __init__(self, filters_out, u_temporal=1, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters_out = filters_out
        self.spatial_block = SpatialShiftConvBlock(filters_out)
        self.temporal_block = TemporalShiftConvBlock(filters_out, u=u_temporal)
        self.residual_conv = None 

    def build(self, input_shape):
        # input_shape: (batch, T, N, C_in)
        filters_in = input_shape[-1]
        if filters_in != self.filters_out:
            self.residual_conv = layers.Conv2D(
                self.filters_out, kernel_size=(1,1), padding='same', name=f"{self.name}_res_proj" if self.name else None
            )
        super().build(input_shape)


    def call(self, inputs, training=False):
        residual = inputs
        
        x = self.spatial_block(inputs, training=training)
        x = self.temporal_block(x, training=training)
        
        if self.residual_conv:
            residual = self.residual_conv(inputs)
            
        x += residual
        return x
    
def build_shift_gcn_model(num_classes, max_frames, num_keypoints, num_coordinates,
                            block_configs,
                            initial_bn=True):
    input_shape = (max_frames, num_keypoints, num_coordinates)
    inputs = keras.Input(shape=input_shape, name="input_layer")

    x = inputs
    if initial_bn:
        x = layers.BatchNormalization(name="initial_bn")(x)

    for i, (filters_out, temporal_u) in enumerate(block_configs):
        x = ShiftGCNBlock(filters_out=filters_out, u_temporal=temporal_u, name=f"shift_gcn_block_{i}")(x)

    x = layers.GlobalAveragePooling2D(data_format='channels_last', name="global_avg_pool")(x)
    
    # Classifier part
    x = layers.Dense(128, activation='relu', name="dense_128")(x)
    x = layers.Dropout(0.5, name="dropout_0.5")(x)
    outputs = layers.Dense(num_classes, activation='softmax', name="output_softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


block_configurations = [
    (64, 2), 
    (128, 2), 
    (256, 2)  
]

def initialize_shift_gcn_model():
    model = build_shift_gcn_model(
        num_classes=NUM_CLASSES,
        max_frames=MAX_FRAMES,
        num_keypoints=NUM_KEYPOINTS,
        num_coordinates=NUM_COORDINATES,
        block_configs=block_configurations
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    model = initialize_shift_gcn_model()
    model_path = "best_model/best_model_shiftgcn.keras"
    model.load_weights(model_path)
    print("Model loaded successfully.")

