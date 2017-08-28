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
import pandas as pd
import numpy as np
import glob
import matplotlib
import cv2
import string
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.decomposition import pca
#from FPCA import pca

# Classifying images are cancerous or non cancerous: 1% of pixels brown
# Later changed to 199 images which are 0 if <20 cells are brown
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

#list = glob.glob('Her2Cells\\trainingset\\*.jpg')
key = pd.read_csv('Her2Cells\\trainingset\\answerKey.txt',sep=' ')
key.columns = ['Name','Cancer']
n_training = len(key)

loc = 'Her2Cells\\trainingset\\'
trainmatrix = np.array([np.array(Image.open(loc+key.Name[i])).flatten() for i in range(n_training)])

"""
fig, ax = plt.subplots(ncols = 2, figsize = (6, 6))
ax.hist(key.Name[key.Cancer == 0], color = 'r', alpha = 0.5, label = "No Cancer", normed = True);
ax.hist(Key.Name[key.Cancer == 1], color = 'b', alpha = 0.5, label = "May Have Cancer", normed = True);

print("Combining %d potentially cancerous cells and %d below-threshold cells for training"
      %(subset_br,subset_bl))

br_train = np.array([np.array(Image.open(cancerlist[i])).flatten() for i in subset_br])
bl_train = np.array([np.array(Image.open(celllist[i])).flatten() for i in subset_bl])
"""

"""
br_train, br_test, bl_train, bl_test = train_test_split(
    X, y, test_size=0.25, random_state=42) #random_state seeds RNG
#don't need to use test_train_split yet because we're only training
"""

# PCA
n_components = 50;
pca = PCA(n_components=n_components,svd_solver='randomized').fit(trainmatrix)
eigenfaces = pca.components_.reshape((3,40,40)) #40x40 pixel images

data_train_pca = pca.transform(trainmatrix)

#n_features = #from PCA

cvc = GridSearchCV(SVC(kernel='rbf',class_weight='balanced',C=0.025, probability=True))
cvc = cvc.fit(data_train_pca,key.Cancer)
cat = ['Cancerous','Not Cancerous']


#cvc.predict(newpoints)
