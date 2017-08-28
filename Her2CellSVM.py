# Her2CellSVM.py by Liana Engie
# Last updated: 2017_8_27
# 
# Working on ML method for classifying images of Her2 antibody
# stained gastric cancer cells
#
# Input: preprocessed TIFF files (cut from NDPI images)
# Output: Proportion of cells classified as cancerous

import os
os.chdir('c:\\Users\cooki\Dropbox\Computer\Science\LaSerena2017')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import string
from scipy import ndimage
from PIL import Image
from sklearn import svm
import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
#from FPCA import pca

cancerlist = glob.glob('Her2Cells\training\cancer\*')
celllist = glob.glob('Her2Cells\training\cancer\*')
subset_br = len(cancerlist)
subset_bl = len(celllist)
print("Combining %d potentially cancerous cells and %d below-threshold cells for training"
      %(subset_br,subset_bl))

#put the cancer and non canerous data together into the same np.array
#X = cancermatrix
#make a target vector 
#y = seq(subset_br*1,subset_bl*1) #i don't actually know the syntax for this
# PCA
n_components = 5;
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

#don't need to use test_train_split yet because we're only training
br_train = np.array([np.array(Image.open(cancerlist[i])).flatten() for i in subset_br])
bl_train = np.array([np.array(Image.open(celllist[i])).flatten() for i in subset_bl])

# Classifying images are cancerous or non cancerous
#answers=open("answerKey.txt","w")
def countBrown(im):
    img = cv2.imread(im)
    lower=np.array([0,0,0],np.uint8)
    upper=np.array([80,255,255],np.uint8)
    dst=cv2.inRange(img,lower,upper)
    brown=cv2.countNonZero(dst)
    print("The number of brown pixels in "+im+": " + str(brown))
    print("    Percent: " + str(100*float(brown)/1600))
    if (100*float(brown)/1600 > 1):
        cancerous="1"
    else:
        cancerous="0"
    print("    Cancerous: " + cancerous)
    answers.write(im + " " + cancerous+"\n")
"""
for i in range(1,85124):
    image="image_x40_z0_"+str(i)+".jpg"
    countBrown(image)
"""

# Images are 40x40 pixel jpgs in HSB
cvc = SVC(kernel='rbf',class_weight='balanced',C=0.025, probability=True)
cvc.fit(br_train,bl_train)
cat = ['Cancerous','Not Cancerous']


#cvc.predict(newpoints)
