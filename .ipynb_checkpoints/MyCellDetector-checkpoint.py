#/usr/bin/python
from __future__ import print_function
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image 
import os, sys
from PyQt5 import QtCore

class MyCellDetector(object):
    def __init__(self):
        self.p = 1
        return None
    
    def read_file(self,file_path: str):
        self.frame = np.array(Image.open(file_path))
#         self.frame = cv2.resize(frame, (self.imW,self.imH))
        return self.frame
     
    def gray_frame(self,frame):
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.gray
        
    def blur_frame(self,frame, kernel=(3,3), filterType = 'gaussian'):
        self.blurWin = kernel
        self.blur = None
        try:
            if filterType =='gaussian':
                self.blur = cv2.GaussianBlur(frame, self.blurWin, 1)
            else:
                self.blur = cv2.medianBlur(frame, self.blurWin[0])
        except:
            print('make sure kernel is composed of odd numbers')
        return self.blur
    
    def get_percentile(self, frame, perc = 90):
        self.pVal_l = np.percentile(frame, 90)
        return self.pVal_l
    
    def mask_frame(self, gray, pVal_l, pVal_u = 255):
        self.mask = (gray>pVal_l)*(gray<pVal_u)
        self.masked = np.zeros_like(gray)
        self.masked[self.mask] = gray[self.mask]
        return (self.mask*1).astype('uint8'), self.masked  
    
    def dilate_frame(self,frame, ker = 5, ite = 1):
        kernel = np.ones((ker,ker), np.uint8)
        self.dilated = cv2.dilate(frame.astype('uint8'), kernel, iterations=ite)
        return self.dilated
    
    
    def erode_frame(self,frame, ker = 5, ite = 1):
        kernel = np.ones([ker,ker], np.uint8)
        self.eroded = cv2.erode(frame.astype('uint8'), kernel, iterations=ite)
        return self.eroded
    

    def countour_frame(self, masked_image,area_mn = 1,area_mx = 1e10, keypoints = {}):
        _, cnts, _ = cv2.findContours(image = masked_image, mode = cv2.RETR_EXTERNAL,\
                                      method = cv2.CHAIN_APPROX_SIMPLE)
        
        
        self.keypoints = keypoints
        ## area constraints
        self.area_mn = area_mn
        self.area_mx = area_mx

        x = np.array([])
        y = np.array([])
        cX = np.array([])
        cY = np.array([])
        h = np.array([])
        w = np.array([])
        C = []
        a = np.array([])
        
        for c in cnts:
            a_ = cv2.contourArea(c)
            # if the contour is too small, ignore it
            if (a_ > self.area_mn) & (a_ < self.area_mx):
                a = np.concatenate([a,np.array([a_])])

                M = cv2.moments(c)
                if M["m00"]<1e-9:
                    M["m00"] = 1e10

                cX_ = int(M["m10"] / M["m00"])
                cY_ = int(M["m01"] / M["m00"])

                cX = np.concatenate([cX,np.array([cX_])])
                cY = np.concatenate([cY,np.array([cY_])])


                (x_, y_, w_, h_) = cv2.boundingRect(c)
                x = np.concatenate([x,np.array([x_])])
                y = np.concatenate([y,np.array([y_])])
                h = np.concatenate([h,np.array([h_])])
                w = np.concatenate([w,np.array([w_])])
                C.append(c)
          
        
        ## not really necessary but useful if you 
        ## only want to get the largest:
        locs = np.argsort(-a)
        x = x[locs].astype('int')
        y = y[locs].astype('int')
        cX = cX[locs].astype('int')
        cY = cY[locs].astype('int')
        h = h[locs].astype('int')
        w = w[locs].astype('int')
        a = a[locs].astype('int')
        self.C = np.array(C)
        self.C = self.C[locs]

        keylen = len(self.keypoints)
        self.keypoints[keylen] = {}

        keypoints[keylen]['num'] = self.C.shape[0]
        self.keypoints[keylen]['contours'] = self.C
        self.keypoints[keylen]['centers'] = np.array([cX, cY]).squeeze().T
        self.keypoints[keylen]['corners'] = np.array([x, y]).squeeze().T
        self.keypoints[keylen]['WH'] = np.array([w, h]).squeeze().T
        self.keypoints[keylen]['areas'] = a
        
        return self.keypoints
        
    def print_keypoints(self, key, displayimage, col= (0,255,0)):
        if len(displayimage.shape)<3:
            self.dispIm = np.repeat(displayimage[:,:,None], repeats = 3,axis = 2)
        else:
            self.dispIm = displayimage.copy()
        
        self.dispImCont = self.dispIm.copy()
        self.dispImCenter = self.dispIm.copy()
        self.dispImBox = self.dispIm.copy()
        n = key['num']
        
        for i in range(key['num']):
            countour = key['contours'][i]
            cXY = key['centers'][i,:]
            XY = key['corners'][i,:]
            WH = key['WH'][i,:]
            self.dispImCont = cv2.drawContours(self.dispImCont, countour, -1, col, 2)
            
                                   
            self.dispImCenter = cv2.circle(self.dispImCenter, (cXY[0], cXY[1]), radius=1, color=col, thickness=5)
            
            
            x = XY[0]
            y = XY[1]
            w = WH[0]
            h = WH[1]
            self.dispImBox =  cv2.rectangle(self.dispImBox,(x,y),(x+w,y+h),col,1)
            
        tmp = self.dispImCenter.copy()
        tmp = tmp.swapaxes(0,1)
        self.dispImCenter = cv2.putText(self.dispImCenter.copy(), "# cells: {}".format(n), (10,100), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
        self.dispImCenter = tmp.copy().swapaxes(0,1)
        
        return self.dispImCont, self.dispImCenter, self.dispImBox