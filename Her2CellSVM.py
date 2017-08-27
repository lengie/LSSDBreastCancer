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
from scipy import ndimage
from PIL import Image
from sklearn import svm
import glob
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
#from FPCA import pca

