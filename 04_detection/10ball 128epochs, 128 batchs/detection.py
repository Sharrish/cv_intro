from json import dumps, load
from numpy import array
from os import environ
from sys import argv


import glob
from os.path import join

import numpy as np
from skimage.io import imread

from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization, Reshape)

from skimage.transform import resize, rotate
from skimage.color import rgb2gray
from sklearn.utils import shuffle
from os import listdir

FAST_TRAIN = True
AXIS_SIZE = 100
EPOCHS_NUMBER = 30
BATCHS_NUMBER = 256
FACEPOINTS_NUMBER = 14
MIN_RAND_ALPHA = -17
MAX_RAND_ALPHA = 17
COUNT_FLIP = 1 if not FAST_TRAIN else 0 
COUNT_ROTATE = 3 if not FAST_TRAIN else 0
COUNT_FRAMING = 0 if not FAST_TRAIN else 0


def resize_img(img, facepoints, new_size):
    new_img = resize(img, (new_size, new_size))
    coeff = (1.0 * new_size) / img.shape[0]
    coeff_reverse = (1.0 * img.shape[0]) / new_size # нужен для обратного масштабирования
    new_facepoints = (facepoints * coeff) if facepoints is not None else None
    return (new_img, new_facepoints, coeff_reverse)


# Зеркальное отражение относительно горизонтальной оси img и точек
def flip_img(img, facepoints):
    new_img = img[:,::-1]
    # Функция numpy.fliplr() отражает массив по горизонтали
    new_facepoints = np.copy(facepoints)
    new_facepoints[:, 0] = img.shape[1] - new_facepoints[:, 0] - 1
    tmp = np.copy(new_facepoints)
    new_facepoints[0, : ] = tmp[3, : ]
    new_facepoints[1, : ] = tmp[2, : ]
    new_facepoints[4, : ] = tmp[9, : ]
    new_facepoints[5, : ] = tmp[8, : ]
    new_facepoints[6, : ] = tmp[7, : ]
    new_facepoints[11, : ] = tmp[13, : ]
    new_facepoints[3, : ] = tmp[0, : ]
    new_facepoints[2, : ] = tmp[1, : ]
    new_facepoints[9, : ] = tmp[4, : ]
    new_facepoints[8, : ] = tmp[5, : ]
    new_facepoints[7, : ] = tmp[6, : ]
    new_facepoints[13, : ] = tmp[11, : ]
    return (new_img, new_facepoints)


# Поворот изображения на небольшой угол
def random_rotate_img(img, facepoints):
    alpha_deg = np.random.randint(MIN_RAND_ALPHA, MAX_RAND_ALPHA)
    new_img = rotate(img, alpha_deg)
    center = img.shape[1] / 2 - 0.5 # как в skimage.transform
    new_facepoints = np.copy(facepoints)
    vec = new_facepoints - center
    alpha_rad = np.radians(alpha_deg)
    rm = [[np.cos(alpha_rad), -np.sin(alpha_rad)],
          [np.sin(alpha_rad), np.cos(alpha_rad)]] # матрица поворота
    rm = np.array(rm)
    vec = (rm.dot(vec.transpose(1, 0))).transpose(1, 0)
    new_facepoints = center + vec
    return (new_img, new_facepoints)


# Кадрирование с сохранением всех точек лица на изображении
def framing_img(img, facepoints, j):
    if (j == 0):
        left_edge = int(0.05 * AXIS_SIZE)
        right_edge = int(0.95 * AXIS_SIZE)
    if (j == 1):
        left_edge = int(0.075 * AXIS_SIZE)
        right_edge = int(0.925 * AXIS_SIZE)
    img = img[left_edge : right_edge, left_edge : right_edge]
    facepoints -= left_edge
    img, facepoints, _ = resize_img(img, facepoints, AXIS_SIZE)
    return (img, facepoints)


# Для предотвращения переобучения и увеличения точности распознавания нейронных
# сетей тренировочная выборка обычно размножается (data_augmentation)
# Функция, подготавливающая данные
def generator_train(train_gt, img_dir, fast_train=False):
    old_total = len(train_gt)
    total = old_total * (1 + COUNT_FLIP + COUNT_ROTATE + COUNT_FRAMING)
    if (fast_train):
        total = 5
    X = np.zeros((total, AXIS_SIZE, AXIS_SIZE))
    y = np.zeros((total, FACEPOINTS_NUMBER, 2))
    for i, (keyy, vallue) in enumerate(train_gt.items()): # имя_файла и лицевые_точки
        img = imread(join(img_dir, keyy))
        img = rgb2gray(img)
        arr1 = vallue[::2] # каждыя четная координата (одномерный массив в строчку)
        arr2 = vallue[1::2] # каждая нечтная координата
        arr1 = np.reshape(arr1, (FACEPOINTS_NUMBER, 1)) # одномерный массив в столбик
        arr2 = np.reshape(arr2, (FACEPOINTS_NUMBER, 1))
        # Функция numpy.reshape() изменяет форму массива без изменения его данных
        facepoints = np.concatenate((arr1, arr2), axis=1)
        img, facepoints, _ = resize_img(img, facepoints, AXIS_SIZE)
        X[i] = img
        y[i] = facepoints
        if not FAST_TRAIN:
            # Зеркальное отражение изображения
            iid = old_total * (1) + i
            X[iid], y[iid] = flip_img(X[i], y[i])
            # Поворот изображения
            for j in range(COUNT_ROTATE):
                iid = old_total * (1 + COUNT_FLIP) + i * COUNT_ROTATE + j
                X[iid], y[iid] = random_rotate_img(X[i], y[i])
            # Кадрирование изображения
            for j in range(COUNT_FRAMING):
                iid = old_total * (1 + COUNT_FLIP + COUNT_ROTATE) + i * COUNT_FRAMING + j
                X[iid], y[iid] = framing_img(X[i], y[i], j)
        if fast_train and i + 1 == total:
            break
    # Нормализация изображений
    mean = X.mean(axis=0, dtype=np.float32)
    dispersion = X.var(axis=0, dtype=np.float32) ** 0.5
    X = (X - mean) / dispersion
    X = np.array(X).reshape(-1, AXIS_SIZE, AXIS_SIZE, 1)
    return X, y


def generator_test(img_dir):
    all_files = sorted(listdir(img_dir))
    total = len(all_files)
    X = np.zeros((total, AXIS_SIZE, AXIS_SIZE))
    all_coeff = np.zeros((total))
    for i, filename in enumerate(all_files):
        img = imread(join(img_dir, filename))
        img = rgb2gray(img)
        img, _, coeff_reverse = resize_img(img, None, AXIS_SIZE)
        X[i] = img
        all_coeff[i] = coeff_reverse
    mean = X.mean(axis=0, dtype=np.float32)
    dispersion = X.var(axis=0, dtype=np.float32) ** 0.5
    X = (X - mean) / dispersion
    X = np.array(X).reshape(-1, AXIS_SIZE, AXIS_SIZE, 1)
    return X, all_files, all_coeff


def get_model():
    model = Sequential() # Создаём модель
    
    model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=(AXIS_SIZE, AXIS_SIZE, 1)))
    model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (2, 2), padding='valid', activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (2, 2), padding='valid', activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (2, 2), padding='valid', activation='relu'))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten()) # сплющивание в вектор чтобы вставить Dense слой

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(FACEPOINTS_NUMBER * 2, activation='relu')) 
    model.add(Reshape((FACEPOINTS_NUMBER, 2))) # Преобразуем вывод в удобную форму.

    # Итак, мы сформировали нашу модель. Теперь нужно подготовить ее к работе:     
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])  
    # loss — это функция ошибки
    # optimizer — используемый оптимизатор (adam в среднем даёт лучший результат) 
    # metrics — метрики, по которым считается качество модели, в нашем случае —
    # это точность (accuracy), то есть доля верно угаданных ответов.
    return model


# Функиця, обучающая модель детектора
def train_detector(train_gt, train_img_dir, fast_train=True):
    epochs = 1 if fast_train else EPOCHS_NUMBER
    batch_sz = 2 if fast_train else BATCHS_NUMBER
    X, y = generator_train(train_gt, train_img_dir, fast_train)
    model = get_model()
    model.fit(X, y, batch_size=batch_sz, epochs=epochs)
    #model.save("facepoints_model.hdf5")


# Функция, проводящая детектирование ключевых точек на изображениях с обученной моделью
def detect(model, test_img_dir):
    X, all_files, all_coeff_reverse = generator_test(test_img_dir)
    pred = model.predict(X)
    ans = {}
    for i in range(0, pred.shape[0], 1):
        pred[i] *= all_coeff_reverse[i]
        ans[all_files[i]] = pred[i].reshape(-1).tolist()
    return ans

'''

def read_csv(filename):
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res

if __name__=="__main__":
  data_dir = "public_tests/00_input/"
  train_dir = join(data_dir, 'train')            
  train_gt = read_csv(join(train_dir, 'gt.csv'))
  train_img_dir = join(train_dir, 'images')

  train_detector(train_gt, train_img_dir, fast_train=False)

'''
