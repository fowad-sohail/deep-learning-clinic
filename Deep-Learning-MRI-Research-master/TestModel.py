'''
Created on Aug 25, 2019

@author: daniel
'''
'''
Created on Dec 26, 2018

@author: daniel
'''

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from DataLoader.DataLoader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from CustomLosses.DiceCoefficient import dice_coef, dice_coef_loss, dice_coef_multilabel, dice_coef_multilabel_loss,dice_coef_ed, dice_coef_et, dice_coef_net





def main():
    num_patients = 1
    W = 128
    H = 128
    normalize = True
    modes = ["flair", "t1", "t1ce", "t2"]
    dataDirectories = ["Data/BRATS_2018/HGG",  "Data/BRATS_2018/LGG"]

    dataLoader = DataLoader(W, H, modes = modes);
    for dataDirectory in dataDirectories:
        dataLoader.loadData(dataDirectory);
    
    
    x_test = dataLoader.getData()
    y_gt = dataLoader.getLabels()

    
    modelDir = "Models/Finalized_Models/unet_2019-01-06-19:50/"

    if normalize:
        mu = np.mean(x_test)
        sigma = np.std(x_test)
        x_test -= mu
        x_test /= sigma
        
    
    
    unet = load_model(modelDir + "model.h5", custom_objects={'dice_coef_ed':dice_coef_ed, 
                                                        'dice_coef_net':dice_coef_net,
                                                        'dice_coef_et': dice_coef_et,
                                                        'dice_coef':dice_coef, 
                                                        'dice_coef_loss':dice_coef_loss,
                                                        'dice_coef_multilabel': dice_coef_multilabel,
                                                        'dice_coef_multilabel_loss' : dice_coef_multilabel_loss})
    
    y_predicted = unet.predict(x_test)


    N = len(y_predicted)
    

    
    for i in range(N):
        fig = plt.figure()
        
        
        pred = y_predicted[i]
        ed_pred = np.rint(pred[:,:,0])
        net_pred = np.rint(pred[:,:,1])
        et_pred = np.rint(pred[:,:,2])
        
        final_pred = np.zeros_like(pred[:,:,0])
        final_pred[ed_pred == 1] = 1
        final_pred[net_pred == 1] = 2
        final_pred[et_pred == 1] = 3


        
        gt = y_gt[i]
        ed_gt = np.rint(gt[:,:,0])
        net_gt = np.rint(gt[:,:,1])
        et_gt = np.rint(gt[:,:,2])
        
        final_gt = np.zeros_like(gt[:,:,0])
        final_gt[ed_gt == 1] = 1
        final_gt[net_gt == 1] = 2
        final_gt[et_gt == 1] = 3

        plt.gray();   
        plt.title("Results", fontweight="bold")
        plt.axis("off")
        
        
        fig.add_subplot(2,6,1)
        plt.imshow(x_test[i,:,:,0])
        plt.axis('off')
        plt.title('FLAIR')
        
        
        fig.add_subplot(2,6,2)
        plt.imshow(x_test[i,:,:,3])
        plt.axis('off')
        plt.title('T2')
        
        fig.add_subplot(2,6,3)
        plt.imshow(ed_gt)
        plt.axis('off')
        plt.title('GT ED')
        
        fig.add_subplot(2,6,4)
        plt.imshow(net_gt)
        plt.axis('off')
        plt.title('GT NET')
        
        fig.add_subplot(2,6,5)
        plt.imshow(et_gt)
        plt.axis('off')
        plt.title('GT ET')
        
        fig.add_subplot(2,6,6)
        plt.imshow(final_gt)
        plt.axis('off')
        plt.title('GT Combined')
        
        
        fig.add_subplot(2,6,7)
        plt.imshow(x_test[i,:,:,1])
        plt.axis('off')
        plt.title('T1')
        
        
        fig.add_subplot(2,6,8)
        plt.imshow(x_test[i,:,:,2])
        plt.axis('off')
        plt.title('T1C')
        
        fig.add_subplot(2,6,9)
        plt.imshow(ed_pred)
        plt.axis('off')
        plt.title('Pred ED')
        
        fig.add_subplot(2,6,10)
        plt.imshow(net_pred)
        plt.axis('off')
        plt.title('Pred NET')
        
        fig.add_subplot(2,6,11)
        plt.imshow(et_pred)
        plt.axis('off')
        plt.title('Pred ET')
        
        fig.add_subplot(2,6,12)
        plt.imshow(final_pred)
        plt.axis('off')
        plt.title('Pred Combined')
        

        plt.show()
    


if __name__ == "__main__":
    main() 
    

    