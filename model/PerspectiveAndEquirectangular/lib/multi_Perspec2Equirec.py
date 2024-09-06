import os
import sys
import cv2
import numpy as np
from model.PerspectiveAndEquirectangular.lib import Perspec2Equirec as P2E


class Perspective:
    def __init__(self, img_array , F_T_P_array, interpolation=cv2.INTER_LINEAR):
        
        assert len(img_array)==len(F_T_P_array)
        
        self.img_array = img_array
        self.F_T_P_array = F_T_P_array

        self.interpolation = interpolation
    

    def GetEquirec(self,height,width, num_channels=3, blending_proportion=0.0):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        merge_image = np.zeros((num_channels, height,width))
        merge_mask = np.zeros((num_channels, height,width))

        for np_img,[F,T,P] in zip (self.img_array,self.F_T_P_array):
            if blending_proportion != 0:
                blending_width = int(np_img.shape[2] * blending_proportion)
                blending_slope = np.tile(np.linspace(0.01, 1, blending_width), (np_img.shape[0], np_img.shape[1], 1))
                
                blending_mask = np.ones_like(np_img)
                blending_mask[..., :blending_width] = blending_slope
                blending_mask[..., np_img.shape[2]-blending_width:] = np.flip(blending_slope, 2)
                blending_mask[:, :blending_width, :] = np.minimum(np.transpose(blending_slope, (0, 2, 1)), blending_mask[:, :blending_width, :])
                blending_mask[:, np_img.shape[2]-blending_width:, :] = np.flip(blending_mask[:, :blending_width, :], 1)
            else:
                blending_mask = np.ones_like(np_img)
                
            img = np_img * blending_mask

            per = P2E.Perspective(blending_mask,F,T,P, interpolation=self.interpolation)        # Load equirectangular image
            blending_mask , mask = per.GetEquirec(height,width, num_channels)   # Specify parameters(FOV, theta, phi, height, width)
            
            per = P2E.Perspective(img,F,T,P, interpolation=self.interpolation)        # Load equirectangular image
            img , mask = per.GetEquirec(height,width, num_channels)   # Specify parameters(FOV, theta, phi, height, width)
            
            merge_image += img
            merge_mask += blending_mask

        merge_mask = np.where(merge_mask==0,1,merge_mask)
        merge_image = (np.divide(merge_image,merge_mask))

        
        return merge_image
        






