import numpy as np

from os import listdir
from os.path import join
from skimage.io import imread

from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                          Dropout, Activation, BatchNormalization)
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam, SGD

from skimage.transform import resize, rotate


FAST_TRAIN = True
AXIS_SIZE = 224
NUMBER_CLASSES = 50
EPOCHS_NUMBER = 20
BATCHS_NUMBER = 256
COUNT_FLIP = 1 if not FAST_TRAIN else 0 
COUNT_ROTATE = 2 if not FAST_TRAIN else 0
COUNT_FRAMING = 1 if not FAST_TRAIN else 0


def resize_img(img):
    new_img = resize(img, (AXIS_SIZE, AXIS_SIZE, 3))
    return new_img


# Зеркалирование по горизонтали
def flip_img(img):
    new_img = np.fliplr(img)
    return new_img


# Поворот на небольшой угол
def random_rotate_img(img):
    alpha_deg = np.random.randint(-17, 17)
    new_img = rotate(img, alpha_deg, mode='edge')
    return new_img


# Кадрирование
def framing_img(img, j):
    if (j == 0):
        left_edge = int(0.05 * AXIS_SIZE)
        right_edge = int(0.95 * AXIS_SIZE)
    if (j == 1):
        left_edge = int(0.075 * AXIS_SIZE)
        right_edge = int(0.925 * AXIS_SIZE)
    new_img = img[left_edge : right_edge, left_edge : right_edge]
    new_img = resize_img(new_img)
    return new_img


def get_model(train_gt, train_img_dir, fast_train=False):
    epochs = 1 if fast_train else EPOCHS_NUMBER
    batch_sz = 1 if fast_train else BATCHS_NUMBER
    resnet_my = ResNet50(
        weights='imagenet',
        include_top=False, # не загружаем часть, отвечающую за классификацию 
        input_shape=(AXIS_SIZE, AXIS_SIZE, 3)) # размер тензоров входных изображений
    
    #resnet_my.trainable = False # сверточную часть обучать не будем
    
    X, y = load_data_to_train(train_gt, train_img_dir, fast_train, resnet_my)
    
    sz = np.prod(resnet_my.layers[-1].output.shape[1:])
    model_class = Sequential()
    model_class.add(Dense(256, input_dim=int(sz)))
    model_class.add(Dropout(0.4))
    model_class.add(Activation('relu'))
    model_class.add(Dense(50))
    model_class.add(Activation('softmax'))
    
    #callbacks = [ModelCheckpoint('birds_model.hdf5', monitor='val_loss', save_best_only=True)]
    
    model_class.compile(loss='categorical_crossentropy', 
                        optimizer=adam(lr=1e-5), 
                        metrics=['accuracy'])
    '''
    history = model_class.fit(
        X,
        y,
        batch_size=batch_sz,
        epochs=epochs,
        validation_split=0.1,
        verbose=2,
        callbacks=callbacks)
    '''
    
    model_class.fit(X, y, epochs=epochs, batch_size=batch_sz, verbose=1)
    
    model_final = Sequential()
    model_final.add(resnet_my)
    model_final.add(Flatten())
    model_final.add(model_class)

    #model_final.save('birds_model.hdf5')
    
    return model_final


def load_data_to_train(train_gt, train_img_dir, fast_train, resnet_my):
    old_total = len(train_gt)
    total = old_total * (1 + COUNT_FLIP + COUNT_ROTATE + COUNT_FRAMING)
    if fast_train:
        total = 5
    sz = np.prod(resnet_my.layers[-1].output.shape[1:])
    sz = int(sz)
    X = (np.zeros((total * sz))).reshape(total, sz)
    y = (np.zeros((total * NUMBER_CLASSES))).reshape(total, NUMBER_CLASSES)
    for i, (filename, class_label) in enumerate(train_gt.items()):
        img = imread(join(train_img_dir, filename))
        img = resize_img(img)
        new_img = image.array_to_img(img)
        X[i] = (resnet_my.predict(np.expand_dims(image.img_to_array(new_img), axis=0))).ravel()
        y[i][int(class_label)] = 1
        if not FAST_TRAIN:
        	# Зеркальное отражение изображения
            iid = old_total * (1) + i
            new_img = flip_img(img)
            new_img = image.img_to_array(image.array_to_img(new_img)) # to keras format
            X[iid] = (resnet_my.predict(np.expand_dims(new_img, axis=0))).ravel()
            y[iid][int(class_label)] = 1
            # Поворот изображения
            for j in range(COUNT_ROTATE):
                iid = old_total * (1 + COUNT_FLIP) + i * COUNT_ROTATE + j
                new_img = random_rotate_img(img)
                new_img = image.img_to_array(image.array_to_img(new_img)) # to keras format
                X[iid] = (resnet_my.predict(np.expand_dims(new_img, axis=0))).ravel()
                y[iid][int(class_label)] = 1
            # Кадрирование изображения
            for j in range(COUNT_FRAMING):
                iid = old_total * (1 + COUNT_FLIP + COUNT_ROTATE) + i * COUNT_FRAMING + j
                new_img = framing_img(img, j)
                new_img = image.img_to_array(image.array_to_img(new_img)) # to keras format
                X[iid] = (resnet_my.predict(np.expand_dims(new_img, axis=0))).ravel()
                y[iid][int(class_label)] = 1
        if fast_train and i + 1 == total:
            break
    return X, y


def load_data_to_test(img_dir):
    all_files = sorted(listdir(img_dir))
    total = len(all_files)
    X = np.zeros((total, AXIS_SIZE, AXIS_SIZE, 3))
    for i, filename in enumerate(all_files):
        img = imread(join(img_dir, filename))
        img = resize_img(img)
        X[i] = image.img_to_array(image.array_to_img(img)) # to keras format
    return X, all_files


# Обучение классификатора на основе предобученной нейросети
# Возвращает: готовую модель нейросети
def train_classifier(train_gt, train_img_dir, fast_train=False):
    model = get_model(train_gt, train_img_dir, fast_train)
    
# Классификация входных изображений при помощи обученной модели
# Возвращает: словарь размером N, keys — имена файлов, 
# values — числа, означающие метку класса (N — количество изображений)
def classify(model, img_dir):
    X, all_files = load_data_to_test(img_dir)
    total = len(all_files)
    ans = {}
    pred = model.predict(X)
    for i in range(0, pred.shape[0], 1):
        ans[all_files[i]] = np.argmax(pred[i])
    return ans

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

  train_classifier(train_gt, train_img_dir, fast_train=FAST_TRAIN)
'''
