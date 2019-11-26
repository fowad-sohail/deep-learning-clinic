'''
Created on Oct 12, 2018

@author: daniel
'''
import keras.backend as K

def dice_coef(y_true, y_pred, smooth=1e-3):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice
    

def dice_coef_loss(y_true, y_pred):
    return -K.log(dice_coef(y_true, y_pred))

def dice_coef_multilabel(N):
    def loss(y_true, y_pred):
        dice = 0
        for index in range(1,N):
            dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
        return dice/N
    return loss

def dice_coef_bg(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,0], y_pred[:,:,:,0])
    return dice


def dice_coef_ed(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,1], y_pred[:,:,:,1])
    return dice

def dice_coef_net(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,2], y_pred[:,:,:,2])
    return dice


def dice_coef_et(y_true, y_pred):
    dice = dice_coef(y_true[:,:,:,3], y_pred[:,:,:,3])
    return dice


def dice_coef_multilabel_loss(N):
    def loss(y_true, y_pred):
        dice = 0
        for index in range(1,N):
            dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
        return -K.log(dice/N)
    return loss






    





