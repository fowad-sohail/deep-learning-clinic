'''
Created on Jan 19, 2019

@author: daniel
'''


from keras.models import Model, Input
from keras.layers import Convolution2D, Activation, BatchNormalization,MaxPooling2D, concatenate
from keras.layers.advanced_activations import ReLU
from keras.layers.convolutional import UpSampling2D


def dilatedInceptionModule(inputs, numFilters = 32): 
    tower_0 = Convolution2D(numFilters, (1,1), padding='same', dilation_rate = (1,1), kernel_initializer = 'he_normal')(inputs)
    tower_0 = BatchNormalization()(tower_0)
    tower_0 = ReLU()(tower_0)
    
    tower_1 = Convolution2D(numFilters, (1,1), padding='same', dilation_rate = (2,2), kernel_initializer = 'he_normal')(inputs)
    tower_1 = BatchNormalization()(tower_1)
    tower_1 = ReLU()(tower_1)
    
    tower_2 = Convolution2D(numFilters, (1,1), padding='same', dilation_rate = (3,3), kernel_initializer = 'he_normal')(inputs)
    tower_2 = BatchNormalization()(tower_2)
    tower_2 = ReLU()(tower_2)
    
    dilated_inception_module = concatenate([tower_0, tower_1, tower_2], axis = 3)
    return dilated_inception_module
    
def createDilatedInceptionUNet(input_shape, n_labels, numFilters=32, output_mode="softmax"):
    inputs = Input(input_shape)
    
    conv1 = dilatedInceptionModule(inputs, numFilters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = dilatedInceptionModule(pool1, 2*numFilters)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = dilatedInceptionModule(pool2, 4*numFilters)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = dilatedInceptionModule(pool3, 8*numFilters)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = dilatedInceptionModule(pool4,16*numFilters)

    up6 = UpSampling2D(size=(2,2))(conv5)
    up6 = dilatedInceptionModule(up6, 8*numFilters)
    merge6 = concatenate([conv4,up6],axis=3)
    
    up7 = UpSampling2D(size=(2,2))(merge6)
    up7 = dilatedInceptionModule(up7, 4*numFilters)
    merge7 = concatenate([conv3,up7],axis=3)
    
    up8 = UpSampling2D(size=(2,2))(merge7)
    up8 = dilatedInceptionModule(up8, 2*numFilters)
    merge8 = concatenate([conv2,up8],axis=3)
    
    up9 = UpSampling2D(size=(2,2))(merge8)
    up9 = dilatedInceptionModule(up9, numFilters)
    merge9 = concatenate([conv1,up9],axis=3)
    
    conv10 = Convolution2D(n_labels, (1,1),  padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv10 = BatchNormalization()(conv10)
    outputs = Activation(output_mode)(conv10)
    
    model = Model(input = inputs, output = outputs)
 
    return model
