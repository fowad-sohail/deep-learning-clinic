'''
Created on Dec 26, 2018

@author: daniel
'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def main():

    modelDir = "Models/Finalized_Models/unet_2019-01-06-19:50/"    
    
    headers = ['epoch', 
               'dice_coef_multilabel', 
               'dice_coef_reg_1', 
               'dice_coef_reg_2', 
               'dice_coef_reg_3',
               'dice_coef_reg_4',
               'loss',
               'val_dice_coef_multilabel',
               'val_dice_coef_reg_1',
               'val_dice_coef_reg_2',
               'val_dice_coef_reg_3',
               'val_dice_coef_reg_4',
               'val_loss']
    
    
    results = pd.read_csv(modelDir + "model_loss_log.csv", names=headers, skiprows=1)
    
    epochs = results['epoch']
    
    
    fig = plt.figure()
    plt.plot(epochs,results['dice_coef_multilabel'], label='average over all segments')
    plt.plot(epochs,results['dice_coef_reg_net'], label='non-enhancing tumor core')
    plt.plot(epochs,results['dice_coef_ed'], label='peritumoral edema')
    plt.plot(epochs,results['dice_coef_et'], label='enhancing tumor')
    plt.title("Training Results", fontweight="bold")
    plt.xlabel('Epoch')
    plt.ylabel('DSC')
    plt.yticks(np.arange(0.5, 1, step=0.1))
    plt.legend()
    
    fig = plt.figure()
    plt.plot(epochs,results['val_dice_coef_multilabel'], label='average over all segments')
    plt.plot(epochs,results['val_dice_coef_net'], label='non-enhancing tumor core')
    plt.plot(epochs,results['val_dice_coef_ed'], label='peritumoral edema')
    plt.plot(epochs,results['val_dice_coef_et'], label='enhancing tumor')
    plt.title("Validation Results",fontweight="bold")
    plt.xlabel('Epoch')
    plt.ylabel('DSC')
    plt.yticks(np.arange(0.5, 1, step=0.1))
    plt.legend()
    
    fig = plt.figure()
    plt.plot(epochs, results['loss'], label = 'training loss')
    plt.plot(epochs,results['val_loss'], label='validation loss')
    plt.title("Loss",fontweight="bold")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yticks(np.arange(0, 1, step=0.1))
    plt.legend()
        
    plt.show()
    
           
        

    


if __name__ == "__main__":
   main() 
    

    