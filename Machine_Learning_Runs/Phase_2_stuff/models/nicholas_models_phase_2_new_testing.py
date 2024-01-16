from keras.models import Model
from keras.layers import Input, MaxPooling3D, GlobalAveragePooling3D, AveragePooling3D, GlobalMaxPooling3D
from keras.layers import Activation, Reshape, Dense, Flatten
from keras.layers import Convolution3D, Conv3DTranspose, UpSampling3D
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.applications import Xception
from keras.applications import VGG19
from keras.regularizers import l1, l1_l2

# "old K1"
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

# The main model we use, the new K1.
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

def new_20X_model_4_with_regularizer(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', activity_regularizer=l1(0.000001), dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', activity_regularizer=l1(0.000001), dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_4_with_regularizer")
    return model

def new_20X_model_4_with_regularizer_V2(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', activity_regularizer=l1(0.0000001), dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', activity_regularizer=l1(0.0000001), dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_4_with_regularizer_V2")
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

def list_model_1(input_shape,
        kernel = 7,
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
    model = Model(inputs=inputs, outputs=outputs, name="list_model_1")
    return model

def list_model_2(input_shape,
        kernel = 7,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="list_model_2")
    return model

def list_model_3(input_shape,
        kernel = 7,
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
    model = Model(inputs=inputs, outputs=outputs, name="list_model_3")
    return model
def list_model_4(input_shape,
        kernel = 7,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="list_model_4")
    return model
def list_model_5(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(200, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(200, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="list_model_5")
    return model
def list_model_6(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(40, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)

    outputs = Convolution3D(40, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(40, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = Conv3DTranspose(40, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)

    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="list_model_6")
    return model
def list_model_7(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)

    outputs = Convolution3D(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = Conv3DTranspose(80, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)

    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="list_model_7")
    return model
def list_model_8(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)

    outputs = Convolution3D(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = Conv3DTranspose(20, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)

    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="list_model_8")
    return model
def list_model_9(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(16, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)

    outputs = Convolution3D(16, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(16, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = Conv3DTranspose(16, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)

    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="list_model_9")
    return model

def list_model_10(input_shape,
        kernel = 5,
        pool_size = (2,2,2)):
    inputs = Input(shape=input_shape)
    outputs = Convolution3D(24, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(inputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)

    outputs = Convolution3D(24, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation = "relu")(outputs)
    #outputs = MaxPooling3D(pool_size = pool_size)(outputs)

    outputs = Conv3DTranspose(24, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)
    outputs = Conv3DTranspose(24, (kernel, kernel, kernel), padding="same", data_format = 'channels_first', dilation_rate=(2,2,2))(outputs)
    outputs = BatchNormalization()(outputs)
    outputs = Activation(activation="relu")(outputs)

    outputs = MaxPooling3D(pool_size = pool_size)(outputs)
    outputs = Flatten()(outputs)
    outputs = Dense(2)(outputs)
    outputs = Activation("softmax")(outputs)
    # print(outputs[0])
    model = Model(inputs=inputs, outputs=outputs, name="list_model_10")
    return model



def new_20X_model_12(input_shape,
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
    model = Model(inputs=inputs, outputs=outputs, name="new_20X_model_12")
    return model