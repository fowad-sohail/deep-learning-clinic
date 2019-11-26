'''
Created on Aug 1, 2018

@author: daniel
'''
import numpy as np
import cv2
import os
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
from keras.utils import np_utils


class DataLoader:
    
    W = None
    H = None
    modes = None
    X = []
    labels = []

    def __init__(self,
                 W = 128, 
                 H = 128,  
                 modes = ["flair", "t1", "t1ce", "t2"],
                 output_segments = ["ed", "net", "et"]):
        self.X = []
        self.labels = []
        self.W = W
        self.H = H
        self.modes = modes
        self.output_segments = output_segments

            
    def clear(self):
        self.X = []
        self.labels = []
        
    def windowIntensity(self, image, min_percent=1, max_percent=99):
        sitk_image = sitk.GetImageFromArray(image)
        sitk_image = sitk.Cast( sitk_image, sitk.sitkFloat32 )
        corrected_image = sitk.IntensityWindowing(sitk_image, 
                                                  np.percentile(image, min_percent), 
                                                  np.percentile(image, max_percent))
        corrected_image = sitk.GetArrayFromImage(corrected_image)
        return corrected_image
        

    def loadData(self, dataDirectory, num_patients = None):
        main_dir = os.listdir(dataDirectory)
        if num_patients is None:
            num_patients = len(main_dir)
        
        for subdir in main_dir[0:num_patients+1]:
            image_dir = dataDirectory + "/" + subdir
            data_dirs = os.listdir(image_dir)
            seg_image = nib.load(image_dir+
                                   "/" + 
                                   [s for s in data_dirs if "seg" in s][0]).get_fdata(caching = "unchanged",
                                                                                      dtype = np.float32)
            
            inds = [i for i in list(range(155)) if np.count_nonzero(seg_image[:,:,i]) > 0]
            foo = {}
            for mode in self.modes:
                for path in data_dirs:
                    if mode in path:
                        if mode is "t1":
                            if "tlce" in path:
                                continue
                        foo[mode] = nib.load(image_dir +
                                          "/" + 
                                          path).get_fdata(caching = "unchanged",
                                                          dtype = np.float32)
                        if len(foo) == len(self.modes):
                            break
            data = [self.processImages(foo, seg_image,i) for i in inds]
            train, labels = zip(*data)
            self.X.extend(train)
            self.labels.extend(labels)



    
    def processImages(self, foo, seg_image, i):
        img = np.zeros((self.W, self.H, len(self.modes)))
        for j,mode in enumerate(self.modes):
            img[:,:,j], rmin, rmax, cmin, cmax = self.zoomOnImage(foo[mode][:,:,i])
        
        img = self.windowIntensity(img)
        seg_img = seg_image[:,:,i]
        seg_img = seg_img[rmin:rmax, cmin:cmax]
        N = len(self.output_segments)
        seg_img[seg_img > N] = N
        seg_img = np_utils.to_categorical(seg_img, N + 1)
        seg_img = cv2.resize(seg_img, 
                     dsize=(self.W, self.H), 
                     interpolation=cv2.INTER_CUBIC)
        seg_img = np.rint(seg_img)
        #self.showData(img, seg_img)

        return img, seg_img
                    
                        
    
    def showData(self, img, seg_img):
        fig = plt.figure()
        plt.gray();   
        N = len(self.output_segments)
        fig.add_subplot(1,N + 1 ,1)
        
        plt.imshow(img[:,:,0])
        plt.axis('off')
        for i in range(1, N + 1):
            fig.add_subplot(1,N+1,i+1)
            plt.imshow(seg_img[:,:,i])
            plt.axis('off')
            plt.title(self.labels[i - 1])
            plt.gray()
        

        plt.show()
        

    def zoomOnImage(self, image):
        rmin,rmax, cmin, cmax = self.bbox(image)
        image = image[rmin:rmax, cmin:cmax]
        resized_image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_CUBIC)
        return resized_image, rmin, rmax, cmin, cmax
    


    def bbox(self, img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin,rmax, cmin,cmax
    
    def getData(self):
        return np.array(self.X)
    
    def getLabels(self):
        return np.array(self.labels)


    


 
