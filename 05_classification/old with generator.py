# Тут надо преобразовывать к keras image format

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


import matplotlib.pyplot as plt

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from skimage.transform import resize, rotate


AXIS_SIZE = 224
NUMBER_CLASSES = 50
EPOCHS_NUMBER = 5
BATCHS_NUMBER = 128
IMG_BATCHS_NUMBER = 25


def get_model():
    resnet_my = ResNet50(
        weights='imagenet',
        include_top=False, # не загружаем часть, отвечающую за классификацию 
        input_shape=(AXIS_SIZE, AXIS_SIZE, 3) # размер тензоров входных изображений
        )
    lr=5 * 10 ** -5
    decay=10 ** -7
    w_reg=5 * 10 ** -5
    
    #resnet_my.trainable = False # сверточную часть обучать не будем
    
    model = Sequential()
    model.add(resnet_my)
    model.add(Flatten())
    model.add(Dense(350))
    model.add(Activation('relu'))
    model.add(Dropout(0.35))
    model.add(Dense(50))
    model.add(Activation('softmax')) # softmax лучше всего подходит для классификации
    
    print(model.summary())
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer=adam(lr=1e-5),
                  metrics=['accuracy'])
    
    return model


def load_data_to_train(train_gt, train_img_dir, fast_train=False):
    total = len(train_gt)
    if (fast_train):
        total = 5
    X = np.zeros((total, AXIS_SIZE, AXIS_SIZE, 3))
    y = (np.zeros((total * NUMBER_CLASSES))).reshape(total, NUMBER_CLASSES)
    # Каждая строка вида [0, 0, 1, 0, 0, ..., 0], где 1 указывает на номер класса
    for i, (filename, class_label) in enumerate(train_gt.items()):
        img = imread(join(train_img_dir, filename)) 
        X[i] = resize(img, (AXIS_SIZE, AXIS_SIZE, 3))
        y[i][int(class_label)] = 1.0
        if fast_train and i + 1 == total:
            break
    print(y)
    X = preprocess_input(X)
    return X, y


def load_data_to_test(img_dir):
    all_files = sorted(listdir(img_dir))
    total = len(all_files)
    X = np.zeros((total, AXIS_SIZE, AXIS_SIZE, 3))
    for i, filename in enumerate(all_files):
        img = imread(join(img_dir, filename))
        X[i] = resize(img, (AXIS_SIZE, AXIS_SIZE, 3))
    return X, all_files


# Обучение классификатора на основе предобученной нейросети
# Возвращает: готовую модель нейросети
def train_classifier(train_gt, train_img_dir, fast_train=False):
    X, y = load_data_to_train(train_gt, train_img_dir, fast_train)
    model = get_model()
    if fast_train:
        model.fit(X, y, batch_size=2, epochs=1)
    else:
        total = len(train_gt)
        
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=15, # max угол поворота img
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest') # заполение пикселей за пределами(aaaaaaaa|abcd|dddddddd)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
        

        train_generator = train_datagen.flow(
            X_train,
            y_train,
            batch_size=IMG_BATCHS_NUMBER, # размер выборки(число прочитанных изображений за 1 раз)
            shuffle=True,
            save_to_dir=None)
        
        model.fit_generator(
            train_generator,
            steps_per_epoch = 2500, # используем каждое изображение ровно один раз
            epochs=EPOCHS_NUMBER,
            verbose=1, # информация
            validation_data=(X_val, y_val)) # данные проверки
        model.save('birds_model.hdf5')
        
    
# Классификация входных изображений при помощи обученной модели
# Возвращает: словарь размером N, keys — имена файлов, 
# values — числа, означающие метку класса (N — количество изображений)
def classify(model, img_dir):
    X, all_files = load_data_to_test(img_dir)
    total = len(all_files)
    ans = {}
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow(
        X,
        batch_size=IMG_BATCHS_NUMBER,
        shuffle=None,
        save_to_dir=None)
    pred = model.predict_generator(
        test_generator,
        steps = total // IMG_BATCHS_NUMBER
    )
    for i in range(0, pred.shape[0], 1):
        print(pred[i])
        ans[all_files[i]] = np.argmax(pred[i])
    print(ans)
    return ans
    #scores = model.evaluate_generator(test_generator, total // IMG_BATCHS_NUMBER)
    #print("Точность на тестовых данных: %.2f%%" % (scores[1]*100))
    

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

