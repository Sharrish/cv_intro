import numpy as np

from os import listdir
from os.path import join
from skimage.io import imread

from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation,
                          BatchNormalization)
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam, SGD

from keras.regularizers import l2
from keras.models import Model





import numpy as np
import os
from skimage.transform import resize
from skimage.io import imread

#!/usr/bin/env python3
from keras.preprocessing import image
import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD

import PIL
import os
from os.path import join

def preprocess_img(img_path):
    img = image.load_img(img_path)
    w, h = img.size
    p = min(w, h) / 224
    w = int(w / p)
    h = int(h / p)
    img = img.resize((w, h))
    img2 = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    img = img.crop((0,0,224,224))
    img2 = img2.crop((0,0,224,224))
    x1 = image.img_to_array(img)
    x2 = image.img_to_array(img2)
    return x1, x2

def train_classifier(train_gt, train_img_dir, fast_train=True):
    res_net = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))

    filenames = list(train_gt.keys())
    N = len(filenames)
    if fast_train:
        y_train = np.zeros(N*50).reshape(N,50)
        X_train = []

        for i in range(N):
            filename = filenames[i]
            class_id = train_gt[filename]
            y_train[i][class_id] = 1
            img_path = join(train_img_dir, filename)
            x1, x2 = preprocess_img(img_path)
            X_train.append(x1)
        X_train = np.array(X_train)

        model = Sequential()
        model.add(res_net)
        model.add(Flatten())
        model.add(Dense(350))
        model.add(Dropout(0.25))
        model.add(Activation('relu'))
        model.add(Dense(50))
        model.add(Activation('softmax'))

        sgd = SGD(lr=0.02, momentum=0.2, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=1, verbose=1, batch_size=64)

        model.save('birds_model.hdf5')
        return model
    else:
        y_train = np.zeros(2*N*50).reshape(2*N,50)
        X_train = []

        for i in range(N):
            filename = filenames[i]
            class_id = train_gt[filename]
            y_train[2*i][int(class_id)] = 1
            y_train[2*i+1][int(class_id)] = 1
            img_path = join(train_img_dir, filename)
            x1, x2 = preprocess_img(img_path)
            #kkk = res_net.predict(np.expand_dims(x1, axis=0))
            #print(kkk.size)
            X_train.append(res_net.predict(np.expand_dims(x1, axis=0)))
            X_train.append(res_net.predict(np.expand_dims(x2, axis=0)))
        X_train = np.array(X_train).reshape(2*N, 100352)

        model_top = Sequential()
        model_top.add(Dense(350, input_dim=100352))
        model_top.add(Dropout(0.25))
        model_top.add(Activation('relu'))
        model_top.add(Dense(50))
        model_top.add(Activation('softmax'))
        sgd = SGD(lr=0.02, momentum=0.2, nesterov=False)
        model_top.compile(loss='categorical_crossentropy',\
                          optimizer=sgd, metrics=['accuracy'])
        
        model_top.fit(X_train, y_train, epochs=15, verbose=1, batch_size=64)
        
        model = Sequential()
        model.add(res_net)
        model.add(Flatten())
        model.add(model_top)
        model.save('birds_model.hdf5')
        return model

def classify(model, test_img_dir):
    preds = {}
    for filename in os.listdir(test_img_dir):
        if filename.endswith(".jpg"):
            img_path = join(test_img_dir,filename)
            x = preprocess_img(img_path)[0]
            x = np.expand_dims(x, axis=0)
            preds[filename] = int(model.predict(x).ravel().argmax())
        else:
            continue
    return preds
  
  
  
'''  
def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            from numpy import array
            coords = array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res

if __name__=="__main__":
  data_dir = "public_tests/00_input/"
  train_dir = join(data_dir, 'train')            
  train_gt = read_csv(join(train_dir, 'gt.csv'))
  train_img_dir = join(train_dir, 'images')

  train_classifier(train_gt, train_img_dir, fast_train=False)
'''
