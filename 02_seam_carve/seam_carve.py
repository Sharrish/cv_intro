import numpy as np
import math


# Вычисление Y компоненты:
def get_Y(image):
    return image.dot(np.array([0.299, 0.587, 0.114])) # произведение матриц


# Вычисление градиента:
def get_grad(Y):
    height = Y.shape[0]
    width = Y.shape[1]
    dx = np.zeros((height, width))
    dx[ : , 0] = Y[ : , 1] - Y[ : , 0]
    dx[ : , 1 : -1] = Y[ : , 2 : ] - Y[ : , : - 2]
    dx[ : , -1] = Y[ : , -1] - Y[ : , -2]
    dy = np.zeros((height, width))
    dy[0, : ] = Y[1, : ] - Y[0, : ]
    dy[1 : -1, : ] = Y[2 : , : ] - Y[ : - 2, : ]
    dy[-1, : ] = Y[-1, : ] - Y[-2, : ]
    return np.sqrt(dx ** 2 + dy ** 2)


# Построение матрицы для нахождения шва с минимальной энергией (ДП)
def get_dp(grad):
    height = grad.shape[0]
    width = grad.shape[1]
    dp = np.copy(grad)
    for y in range(1, height, 1):
        for x in range(0, width, 1):
            mn = dp[y - 1, x]
            if x > 0 and mn > dp[y - 1, x - 1]:
                mn = dp[y - 1, x - 1]
            if x < width - 1 and mn > dp[y - 1, x + 1]:
                mn = dp[y - 1, x + 1]
            dp[y][x] += mn
    return dp


# Находим шов с минимальной энергией
def get_seam(dp):
    height = dp.shape[0]
    width = dp.shape[1]
    seam = np.zeros(height, dtype=int)
    id_mn = np.argmin(dp[-1]) # индекс min элемента в последней строке
    for i in range(height - 1, -1, -1):
        mn = dp[i, id_mn]
        tmp_id = id_mn
        if (id_mn > 0 and mn >= dp[i, id_mn - 1]):
            tmp_id = id_mn - 1
            mn = dp[i, id_mn - 1]
        if (id_mn < width - 1 and mn > dp[i, id_mn + 1]):
            tmp_id = id_mn + 1
            mn = dp[i, id_mn + 1]
        id_mn = tmp_id
        seam[i] = id_mn
    return seam


def seam_carve(image, mode, mask=None): # mask - (Опциональный аргумент)
    mode = mode.split(' ')
    # Реализация для горизонтального изменения изображения
    if mode[0] == 'vertical': # (для вертикального транспонируем матрицу)
        if mask is not None:
            mask = mask.T
        image = image.transpose(1, 0, 2) # оси будут расставлены в указанном порядке
    height = image.shape[0]
    width = image.shape[1]
    new_mask = None
    if mode[1] == 'shrink':
        new_image = np.zeros((height, width - 1, 3))
        if mask is not None:
            new_mask = np.zeros((height, width - 1))
    if mode[1] == 'expand':
        new_image = np.zeros((height, width + 1, 3))
        if mask is not None:
            new_mask = np.zeros((height, width + 1))
    mask_seam = np.zeros((height, width)) # хранит шов в ответе
    BOOST = height * width * 256 # заведомо большая величина из условия
    Y = get_Y(image)
    grad = get_grad(Y)
    if mask is not None:
        grad += BOOST * mask
    dp = get_dp(grad)
    seam = get_seam(dp)
    for i in range(0, height, 1):
        j = seam[i]
        mask_seam[i, j] = 1
        if mode[1] == 'shrink':
            new_image[i] = np.delete(image[i], j, axis=0)
            if mask is not None:
                new_mask[i] = np.delete(mask[i], j)
        if mode[1] == 'expand':
            pix = image[i, j]
            new_image[i] = np.insert(image[i], j + 1, image[i, j], axis=0)
            if mask is not None:
                new_mask[i] = np.insert(mask[i], j + 1, mask[i, j])
    if mode[0] == 'vertical': # транспонируем все обратно
        new_image = new_image.transpose(1, 0, 2)
        if mask is not None:
            new_mask = new_mask.T
        mask_seam = mask_seam.T
    return (new_image, new_mask, mask_seam)