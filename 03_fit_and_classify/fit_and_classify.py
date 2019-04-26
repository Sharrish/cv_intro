import numpy as np
import math
from sklearn import svm
from sklearn.svm import LinearSVC
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

# Вычисление Y компоненты:
def get_Y(image):
    return image.dot(np.array([0.299, 0.587, 0.114])) # произведение матриц


# Вычисление градиента изображения по
# горизонтальному и вертикальному направлению:
def get_grad(Y):
    dx = np.zeros((Y.shape[0], Y.shape[1]))
    dx[ : , 0] = Y[ : , 1] - Y[ : , 0]
    dx[ : , 1 : -1] = Y[ : , 2 : ] - Y[ : , : -2]
    dx[ : , -1] = Y[ : , -1] - Y[ : , -2]
    dy = np.zeros((Y.shape[0], Y.shape[1]))
    dy[0, : ] = Y[1, : ] - Y[0, : ]
    dy[1 : -1, : ] = Y[2 : , : ] - Y[ : -2, : ]
    dy[-1, : ] = Y[-1, : ] - Y[-2, : ]
    return (dy, dx)


# Вычисление для каждого пикселя изображения
# величины и направления градиента
def valGrad_orientation(Y):
    (dy, dx) = get_grad(Y)
    orientation = (np.arctan2(dy, dx) + np.pi) / 2 # [0, n.pi] 
    return (np.hypot(dy, dx), orientation)


# Нормировка вектора v
def normalize(v, eps = 1e-8):
        return (v / np.sqrt(np.sum(v ** 2) + eps ** 2))


# Извлечение признаков на основе гистограмм ориентированных градиентов
def extract_hog(img):
    # Разбиение на ячейки и для каждой ячейки строится 
    # гистограмма направлений c binCount корзин(направлений)
    # cellRows × cellCols - размеры ячеек разбиения (в пикселях)
    cellRows = 8
    cellCols = 8
    # Далал и Триггс обнаружили, что беззнаковый градиент совместно с девятью каналами
    # гистограммы дает лучшие результаты при распознавании людей (Википедия)
    # Экспериментально было выявлено [1], что оптимальное качество распознавания достигается
    # при количестве бинов порядка восьми. (статья МФТИ)
    binCount = 9 # binCount - число корзин (направлений)
    img = get_Y(img)
    '''
    Здесь предлагается подход к распознаванию изображений, основанный 
    на построение гистограмм по ячейкам постоянного размера.
    Недостаток: изображения различного разрешения должны быть приведены к 
    некоторому общему среди выборки изображений разрешению
    '''
    img = resize(img, (86, 86))
    valGrad, orientation = valGrad_orientation(img)
    # hight_in_cell, width_in_cell - высота, ширина таблицы для хранения ячеек
    hight_in_cell = int(math.ceil(img.shape[0] / cellRows)) # округлили к большему
    width_in_cell = int(math.ceil(img.shape[1] / cellCols))
    hist = np.zeros((binCount, hight_in_cell, width_in_cell))
    # Проходим по всем пикселям
    for y in range(0, img.shape[0], 1):
        for x in range(0, img.shape[1], 1):
            # y1, x1 - кординаты ячейки, в которую попал пиксель
            y1 = y // cellRows
            x1 = x // cellCols
            # определяем корзину, округлили к меньшему
            basket = int(math.floor((orientation[y, x] * binCount) / np.pi))
            if (basket < binCount):
                hist[basket, y1, x1] += valGrad[y, x]
            else:
                hist[0, y1, x1] += valGrad[y, x]
    histograms = np.array([]) # итоговый массив "гистограмм блоков"
    # Прореживание. (скользящим окном(блоком))
    # blockRowCells × blockColCells - размеры блоков(прореживающего окна) (в ячейках)
    iterations = [(2, 2), (8, 8), (16, 16)]
    for i in enumerate(iterations):
        (blockRowCells, blockColCells) = i[1]
        for y in range(0, hight_in_cell - blockRowCells + 1, blockRowCells // 2):
            for x in range (0, width_in_cell - blockColCells + 1, blockColCells // 2):
                # Функция numpy.ravel() возвращает сжатый до одной оси массив
                block = normalize(hist[ : , y : y + blockRowCells, x : x + blockColCells].ravel())
                histograms = np.concatenate([histograms, block])
    #print(histograms.size)
    return histograms


def fit_and_classify(train_features, train_labels, test_features):
    """
    Хорошие константы: 2.25, 3.5, 50.1, 7.0
    """
    XX, YY = shuffle(train_features, train_labels)
    
    '''
    constant = 2.25
    clf = LinearSVC(C=constant, max_iter=10000)
    scores = cross_val_score(clf, XX, YY, cv=7)
    print("Accuracy: %0.6f (+/- %0.2f)" % (scores.mean(), scores.std() * 2), "При константе:", constant)
    '''

    clf = LinearSVC(C=2.25, max_iter=10000)
    clf.fit(XX, YY)
    return clf.predict(test_features)
