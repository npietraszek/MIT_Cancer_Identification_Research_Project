from keras.models import Model
from keras.layers import Input, MaxPooling3D, GlobalAveragePooling3D, AveragePooling3D, GlobalMaxPooling3D
from keras.layers.core import Activation, Reshape, Dense, Flatten
from keras.layers.convolutional import Convolution3D, Conv3DTranspose, UpSampling3D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout
from keras.applications import Xception
from keras.applications import VGG19
from keras.regularizers import l1, l1_l2
def find_the_max(the_outputs):
    max_index = 0
    max = 0
    for x in range(3):
        if max < the_outputs[x]:
            max = x
            max_index = x
    for x in range(3):
        the_outputs[x] = 0
    the_outputs[max_index] = 1
    return the_outputs


# From this point on, no more trying random things!
# Types of data: spheres, cubes, mixes of cubes and spheres?
# model 10: Has dilated Convolution networks to better grasp the key features of identifying spheres and cubes
# Also need pooling because that helps with learning a lot.
# 2 Pooling layers: One at MaxPooling3D, one at the end being GlobalAveragePooling3D


# "K1"
def the_model_9(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(16, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(16, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="model_9")
    return model

# "K1"
def the_model_9_without_bottom(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(16, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(16, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="model_9")
    return model

def new_20X_model_1(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_1")
    return model

def new_20X_model_2(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(40, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(40, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_2")
    return model

def new_20X_model_3(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(16, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(16, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_3")
    return model

def new_20X_model_4(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_4")
    return model

def new_20X_model_5(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(120, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(120, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_5")
    return model

# A more typical convolution then deconvolution network.
def new_20X_model_6(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(16, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    outputs = MaxPooling3D(pool_size=pool_size)(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(16, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(16, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    outputs = UpSampling3D(size=(2, 2, 2))(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(16, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = UpSampling3D(size = (2,2,2))(outputs)

    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_6")
    return model

# A more typical convolution then deconvolution network.
def new_20X_model_7(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    outputs = MaxPooling3D(pool_size=pool_size)(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Dropout(rate = 0.2)(outputs)

    outputs = Conv3DTranspose(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    outputs = UpSampling3D(size=(2, 2, 2))(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = UpSampling3D(size = (2,2,2))(outputs)

    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_7")
    return model



def new_20X_model_8(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Dropout(rate = 0.2)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_8")
    return model

def new_20X_model_9(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(60, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(60, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_9")
    return model

def new_20X_model_10(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(40, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(40, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_10")
    return model

# A more typical convolution then deconvolution network.
def new_20X_model_11(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    outputs = MaxPooling3D(pool_size=pool_size)(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    outputs = UpSampling3D(size=pool_size)(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = UpSampling3D(size = pool_size)(outputs)

    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_11")
    return model

def new_20X_model_12(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    outputs = MaxPooling3D(pool_size=pool_size)(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_12")
    return model


def new_20X_model_13(input_shape,
                    kernel=5,
                    pool_size=(2, 2, 2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(16, (kernel, kernel, kernel), padding="same", data_format='channels_first',
                            dilation_rate=(2, 2, 2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    # outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(16, (kernel, kernel, kernel), padding="same", data_format='channels_first',
                            dilation_rate=(2, 2, 2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size=pool_size)(outputs)
    outputs = Dropout(rate = 0.5)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_13")
    return model

def new_20X_model_14(input_shape,
                    kernel=5,
                    pool_size=(2, 2, 2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(16, (kernel, kernel, kernel), padding="same", data_format='channels_first',
                            dilation_rate=(2, 2, 2), activity_regularizer=l1(0.01))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    # outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(16, (kernel, kernel, kernel), padding="same", data_format='channels_first',
                            dilation_rate=(2, 2, 2), activity_regularizer=l1(0.01))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size=pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_14")
    return model
def new_20X_model_15(input_shape,
                    kernel=5,
                    pool_size=(2, 2, 2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format='channels_first',
                            dilation_rate=(2, 2, 2), activity_regularizer=l1(0.1))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    # outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format='channels_first',
                            dilation_rate=(2, 2, 2), activity_regularizer=l1(0.1))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size=pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_15")
    return model
def new_20X_model_16(input_shape,
                    kernel=5,
                    pool_size=(2, 2, 2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format='channels_first',
                            dilation_rate=(2, 2, 2), activity_regularizer=l1(0.0001))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    # outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format='channels_first',
                            dilation_rate=(2, 2, 2), activity_regularizer=l1(0.0001))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size=pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_16")
    return model

def new_20X_model_17(input_shape,
                    kernel=5,
                    pool_size=(2, 2, 2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format='channels_first',
                            dilation_rate=(2, 2, 2), activity_regularizer=l1(0.00001))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    # outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format='channels_first',
                            dilation_rate=(2, 2, 2), activity_regularizer=l1(0.00001))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size=pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_17")
    return model
def new_20X_model_18(input_shape,
                    kernel=5,
                    pool_size=(2, 2, 2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format='channels_first',
                            dilation_rate=(2, 2, 2), activity_regularizer=l1(0.000001))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    # outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format='channels_first',
                            dilation_rate=(2, 2, 2), activity_regularizer=l1(0.000001))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size=pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_17")
    return model

def new_20X_model_19(input_shape,
                    kernel=5,
                    pool_size=(2, 2, 2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format='channels_first',
                            dilation_rate=(2, 2, 2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    # outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format='channels_first',
                            dilation_rate=(2, 2, 2), activity_regularizer=l1_l2(0.0001))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size=pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_17")
    return model

# Failed completely since K1's output is just the prediction...
# But this was a helpful experiment for understanding how to implement transfer learning
def transfer_learning_model_1(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    base_model = the_model_9(input_shape)
    for layer in base_model.layers: layer.trainable = False
    base_model.load_weights(
        r"D:\MIT_Tumor_Identifcation_Project_Stuff\Phase_1_stuff\Machine learning stuff\Best_Models_So_Far\Harshly_balanced\Current Best\7_14_20_new_run_1_(model_9,harsh_balance,dataset_6,xy_rotate)\Superloop_run_19\7_14_20_Test1_balanced_xy_DNN.56.hdf5")


    outputs = base_model.output
    outputs = Convolution3D(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="transfer_learning_model_1")
    return model

# Transfer learning also does not work for keras.application models since the input_shape is required to be 3 dimensions,
# whereas we have 4 due to our multiple image types.
# In order to use keras applications models, we would need to perform an ensemble approach.
def transfer_learning_model_2(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    base_model = Xception(include_top = False,input_shape = input_shape)(inputs)
    for layer in base_model.layers: layer.trainable = False
    outputs = base_model.output
    outputs = Convolution3D(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    outputs = MaxPooling3D(pool_size=pool_size)(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = UpSampling3D(size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="transfer_learning_model_2")
    return model
