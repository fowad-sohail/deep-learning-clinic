'''
Created on Aug 17, 2019

@author: daniel
'''
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from datetime import datetime

from UNet.createUNet import createUNet
from InceptionUNet.createtInceptionUNet import createInceptionUNet
from DataLoader.DataLoader import DataLoader
from keras.callbacks import CSVLogger
from CustomLosses.DiceCoefficient import dice_coef_multilabel, dice_coef_multilabel_loss, dice_coef_ed, dice_coef_net, dice_coef_et
from keras.optimizers import Adam
from Utils.HardwareHandler import HardwareHandler
import tensorflow as tf
from keras.utils.training_utils import multi_gpu_model



def main():
    labels = ["ed", "net", "et"]
    N = len(labels)
    modes = ["flair","t1", "t1ce", "t2"]
    W = 128
    H = 128
    
    num_epochs = 100
    batch_size = 16
    validation_split = 0.1
    now = datetime.now()
    date_string = now.strftime('%Y-%m-%d-%H:%M')
    dataDirectories = ["Data/BRATS_2018/HGG",  "Data/BRATS_2018/LGG"]

    dataLoader = DataLoader(W, H, modes = modes);
    for dataDirectory in dataDirectories:
        dataLoader.loadData(dataDirectory);
        
    X = dataLoader.getData()
    Y = dataLoader.getLabels()

    input_shape = (W, H, len(modes))
        
    optimizer = Adam()
    
    numGPUs = HardwareHandler().getAvailableGPUs()
    model_directory = "Models/unet_" + date_string 
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
            
    log_info_filename = 'model_loss_log.csv'
    csv_logger = CSVLogger(model_directory + '/' + log_info_filename, append=True, separator=',')
    if numGPUs > 1:
        with tf.device('/cpu:0'):
            unet_to_save = createUNet(input_shape, N + 1, output_mode = "softmax")
        unet = multi_gpu_model(unet_to_save, numGPUs)
    else:
        unet = createUNet(input_shape, N + 1, output_mode = "softmax")
        
    metrics = [dice_coef_multilabel(N)]
    if "ed" in labels:
        metrics.append(dice_coef_ed)
    if "net" in labels:
        metrics.append(dice_coef_net)
    if "et" in labels:
        metrics.append(dice_coef_et)
        
    loss = dice_coef_multilabel_loss(N)    
    unet.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    unet.fit(X, Y, batch_size = batch_size, epochs=num_epochs, validation_split=validation_split, callbacks=[csv_logger])
    
    if numGPUs > 1:
        unet_to_save.save(model_directory + '/model.h5')
    else:
        unet.save(model_directory + '/model.h5')

if __name__=="__main__":
    main()
    exit()

    
