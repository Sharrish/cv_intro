import numpy as np # Это выражение позволяет нам получать доступ к numpy объектам используя np.X вместо numpy.X.
from skimage.transform import rescale


# Среднеквадратичное отклонение для изображений i1 и i2
# Для нахождения оптимального сдвига нужно взять минимум по всем сдвигам.
def mse(i1, i2):
    return ((i1 - i2) ** 2).sum() / (i1.shape[0] * i1.shape[1])


# Нормализованная кросс-корреляция для изображений i1 и i2:
# Для нахождения оптимального сдвига нужно взять максимум по всем сдвигам.
def cross_corr(i1, i2): # работает дольше, чем mse
    return (i1 * i2).sum() / np.sqrt((i1 ** 2).sum() * (i2 ** 2).sum())


# использование метрики, mse - flag == 1, cross_corr - flag == 2:
def metric(i1, i2, metric_example, offset, flag):
    # Для того, чтобы совместить два изображения, будем сдвигать одно изображение
    # относительно другого в некоторых пределах, например, от −15 до 15 пикселей
    ans = np.inf
    if (flag == 2):
        ans = -ans
    for y in range(-offset, offset + 1, 1): # лучше сначала по высоте
        for x in range(-offset, offset + 1, 1):
            # Функция numpy.roll(a, shift, axis = None) циклическое смещение элементов
            # массива вдоль указанной оси. a - массив, shift - смещение, axis - ось
            tmp_im1 = i1[max(-y, 0) : i1.shape[0] - y, max(-x, 0) : i1.shape[1] - x]
            tmp_im2 = i2[max(y, 0) : i2.shape[0] + y, max(x,0) : i2.shape[1] + x]
            res = metric_example(tmp_im1, tmp_im2)
            if ((flag == 1 and ans > res) or (flag == 2 and ans < res)):
                ans = res
                best_result = (y, x)
    return best_result


# пирамида изображений, mse - flag == 1, cross_corr - flag == 2:
def pyramid(i1, i2, metric_example, flag):
    if (i1.shape[0] <= 500 and i1.shape[1] <= 500):
        # размер, при котором больше не уменьшаем, а просто считаем смещение
        return metric(i1, i2, metric_example, 15, flag)
    else:
        # skimage.transform.rescale - Выполняет интерполяцию для изменения масштаба изображений.
        tmp_im1 = rescale(i1, 0.5)  # уменьшаем изображения в 2 раза
        tmp_im2 = rescale(i2, 0.5)
        offset = 2 * np.array(pyramid(tmp_im1, tmp_im2, metric_example, flag))
        tmp_im1 = i1[max(-offset[0], 0) : i1.shape[0] - offset[0], max(-offset[1], 0) : i1.shape[1] - offset[1]]
        tmp_im2 = i2[max(offset[0], 0) : i2.shape[0] + offset[0], max(offset[1],0) : i2.shape[1] + offset[1]]
        # 1 подбиралась для прохода всех тестов тесты (чем меньше, тем лучше для скорости)
        offset_end = metric(tmp_im1, tmp_im2, metric_example, 1, flag) # последний сдвиг больших изображений
        return offset + offset_end


def align(grey_image, green):
    # атрибут изображения shape, покажет размерность NumPy массива (высота, ширина, число каналов цвета)
    height = grey_image.shape[0] // 3
    width = grey_image.shape[1]
    
    h_frame = int(height * 0.05) # размер рамки пленки для каждого канала по высоте
    w_frame = int(width * 0.05) # размер рамки пленки для каждого канала по ширине
    
    # деление на каналы и обрезка рамки пленки используя срезы NumPy [срез по высоте, срез по ширине]
    blue_image = grey_image[h_frame : height - h_frame, w_frame : width - w_frame]
    green_image = grey_image[height + h_frame : 2 * height - h_frame, w_frame : width - w_frame]
    red_image = grey_image[2 * height + h_frame : 3 * height - h_frame, w_frame : width - w_frame]

    blue = pyramid(blue_image, green_image, mse, 1)
    red = pyramid(red_image, green_image, mse, 1)
    #blue = pyramid(blue_image, green_image, cross_corr, 2)
    #red = pyramid(red_image, green_image, cross_corr, 2)

    # Функция совмещения align должна по точке (g_row, g_col) зеленого канала определить координаты
    # соответствующих ей точек синего и красного каналов: (b_row, b_col),(r_row, r_col).
    g_row, g_col = green
    (b_row, b_col) = (g_row - blue[0] - height, g_col - blue[1])
    (r_row, r_col) = (g_row - red[0] + height, g_col - red[1])
    
    red_image = np.roll(red_image, red[0], axis = 0)
    red_image = np.roll(red_image, red[1], axis = 1)
    blue_image = np.roll(blue_image, blue[0], axis = 0)
    blue_image = np.roll(blue_image, blue[1], axis = 1)
    
    # создание цветного изображения
    im_out = np.dstack((red_image, green_image, blue_image))
    return im_out, (b_row, b_col), (r_row, r_col)