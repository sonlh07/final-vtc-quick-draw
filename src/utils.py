import glob
import numpy as np
import cv2
from scipy import ndimage
import math
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical

train_file_path = 'doodle_data/npy/'
npy_ext = '*.npy'


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def pre_process(img):
    img=255-np.array(img).reshape(28,28).astype(np.uint8)
    (thresh, gray) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    while np.sum(gray[0]) == 0:
        gray = gray[1:]

    while np.sum(gray[:,0]) == 0:
        gray = np.delete(gray,0,1)

    while np.sum(gray[-1]) == 0:
        gray = gray[:-1]

    while np.sum(gray[:,-1]) == 0:
        gray = np.delete(gray,-1,1)

    rows,cols = gray.shape

    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols*factor))
        gray = cv2.resize(gray, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        gray = cv2.resize(gray, (cols, rows))

    colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
    rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
    gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

    shiftx,shifty = getBestShift(gray)
    shifted = shift(gray,shiftx,shifty)
    gray = shifted

    img = gray.reshape(28, 28, 1).astype(np.float32)

    img-= int(33.3952)
    img/= int(78.6662)
    return img


def image_prepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    tv = list(newImage.getdata())  # get pixel values
    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    print(tva)
    return tva


def get_list_train_npy():
    return glob.glob(train_file_path + npy_ext)


def load_data(class_name):
    data = np.load(train_file_path + class_name)
    return data


def prepare_train_data():
    y = list()
    X = list()
    list_names = get_list_train_npy()
    for index, full_name in enumerate(list_names, 0):
        name = full_name.rsplit('/', 1)[1]
        class_data = load_data(name)
        y.extend([index] * len(class_data))
        X.extend(class_data)
    return np.array(X), np.array(y)


def get_data_info(data):
    print('Type', type(data))
    print('Shape', data.shape)


def view_image(image):
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.show()


def view_sample(image):
    plt.imshow(image[:, :, 0], cmap=plt.get_cmap('gray'))
    plt.show()


def get_check_point(saved_path):
    return ModelCheckpoint(saved_path, monitor='loss',
                           verbose=1,
                           save_best_only=True,
                           mode='auto',
                           period=1)


def convert_for_CNN(X, y):
    X = X / 255.0
    X = X.reshape([-1, 28, 28, 1])
    y = to_categorical(y, num_classes=3)
    return X, y


