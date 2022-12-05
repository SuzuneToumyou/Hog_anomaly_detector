#!/usr/bin/python3
# -*- coding: utf-8 -*

import numpy as np
import cv2
import matplotlib.pyplot as plt

import sys
import pathlib as pl

#gen_train_data
from skimage.feature import hog
from skimage import data, exposure

#train
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import pickle

#gen_train_data
image_size_x = 100 #100
image_size_y = 100 #100
reshape_size = 8100 #39200

cell_size = 8 #6
block_size = 3 #10
orientations_number = 9

#train
n_dim = 4
nu = 0.4
g_hosei = 50

train_data = "./train.csv"

weights = "./weights.sav"
weights_pca = "./weights_pca.sav"

def preprocess(image):
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame = cv2.equalizeHist(frame)
    return frame

def gen_train_data(pass_intmp):

    pass_intmp2 = "./" + pass_intmp
    p_temp = pl.Path(pass_intmp2).glob("*.jpg")

    feature = []
    for p in p_temp:

        img =  plt.imread(p)
        img = cv2.resize(img, (image_size_x, image_size_y))
        img =  preprocess(img)
        fd, hog_image = hog(img, orientations=orientations_number, pixels_per_cell=(cell_size, cell_size),cells_per_block=(block_size, block_size), visualize=True)
#       fd, hog_image = hog(img, visualize=True,transform_sqrt=True)
        feature = np.append(feature,fd)
        np.savetxt(train_data, feature.reshape(-1,reshape_size), delimiter=",")
        #
        #plt.imshow(hog_image,cmap=plt.cm.gray)
        #plt.show()

def train():
    x_train = np.loadtxt(train_data,delimiter=",")
    pca = PCA(n_components=n_dim)
    #clf = OneClassSVM(nu=nu, gamma=g_hosei/n_dim)#1/n_dim
    
    clf = OneClassSVM(nu=nu, gamma="auto")
    z_train = pca.fit_transform(x_train)
    clf.fit(z_train)

    pickle.dump(pca, open(weights_pca, "wb"))
    pickle.dump(clf,open(weights,"wb"))

def judgement_data(pass_intmp):
    clf = pickle.load(open(weights,"rb"))
    pca = pickle.load(open(weights_pca, "rb"))
    
    pass_intmp3 = "./" + pass_intmp
    p_temp = pl.Path(pass_intmp3).glob("*.jpg")

    for p in p_temp:
    
        img =  plt.imread(p)
        img = cv2.resize(img, (image_size_x, image_size_y))
        img =  preprocess(img)
        fd, hog_image = hog(img, orientations=orientations_number, pixels_per_cell=(cell_size, cell_size),cells_per_block=(block_size, block_size), visualize=True)
        #fd, hog_image = hog(img, visualize=True,transform_sqrt=True)
        z_feature = pca.transform(fd.reshape(1,reshape_size))
        score = clf.predict(z_feature.reshape(1,n_dim))
        #print(score[0])
        if score[0]== 1:
            print(p,":正常値")
        else :
            print(p,":異常値")
        #plt.imshow(hog_image,cmap=plt.cm.gray)
        #plt.show()

if __name__ == "__main__":

    
    if(len(sys.argv) <= 1):
        print("コマンドライン入力せよ")
        print("0:学習データ作成")
        print("1:学習")
        print("2:判定")
        
    elif (sys.argv[1]=="0"):
        gen_train_data("traindata")
        
    elif (sys.argv[1]=="1"):
        train()
        
    elif (sys.argv[1]=="2"):
        judgement_data("judgement")
        
    elif (sys.argv[1]=="3"):
        gen_train_data("traindata")
        train()
        judgement_data("judgement")
    else:
        print("エラー")