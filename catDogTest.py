# catDogTest.py by Liana Engie
# Last updated: 2017_8_26
# 
# Working on ML method for classifying images of cats and dogs,
# precursor to working on Her2 cancer image classification
#
# Input: Kaggle cats & dogs data
# Output: Table classifying image name and photo subject

import os
os.chdir('c:\\Users\cooki\Dropbox\Computer\Science\LaSerena2017')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from PIL import Image
from sklearn import svm
import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
#from FPCA import pca

# For now, using a subset of 100 cat pictures and 100 dog pictures
catlist = glob.glob('cat_dog_data\training\cats\*')
doglist = glob.glob('cat_dog_data\training\dogs\*')
subset = len(catlist)

width=40
height=40

# cat-dog challenge: normalize to 20x20, cell images will be 40x40

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img
for imgname in catlist:
    img = cv2.imread(imgname)
    trans = transform_img(img,width,height)
    cv2.imwrite(img,trans)
    
for imgname in doglist:
    img = cv2.imread(imgname)
    trans = transform_img(img,width,height)
    cv2.imwrite(img,trans)

cat_matrix=np.array([np.array(Image.open(catlist[i])).flatten() for i in subset])
dog_matrix=np.array([np.array(Image.open(doglist[i])).flatten() for i in subset])

# PCA, but I can't get FPCA package
"""
cV,cS,c_mean = pca(cat_matrix)
c_mean = c_mean.reshape(width,height)   
c_mode = cV[0].reshape(width,height) #changing the eigen vetor back into an image

dV,dS,d_mean = pca(dog_matrix)
d_mean = d_mean.reshape(width,height)
d_mode = dV[0].reshape(width,height) #changing the eigen vetor back into an image

pylab.figure()
pylab.gray()
pylab.imshow(immean)
pylab.figure()
pylab.gray()
pylab.imshow(mode)
pylab.show()
"""

#Using all 200 images as training data

# train a c-support vector classifier and plot the probability
"""
clf = SVC(kernel="linear", C=0.025, probability = True)
clf.fit(c_mode, d_mode)
dog_pred = clf.predict_proba(cat_test)[:, 1]
xs = np.linspace(min(X), max(X), 100)
xs = xs[:, np.newaxis]
probs = clf.predict_proba(xs)[:, 1]
ax[0].plot(xs, probs, label = "Probability of class 1")
ax[0].legend()
ax[1].legend()
"""
`
