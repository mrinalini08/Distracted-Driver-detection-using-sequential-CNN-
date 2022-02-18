
from tqdm import tqdm
from glob import glob
import cv2
import np_utils
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
import timeit

base_dir = '../../../Downloads/distracted_driver_detection_folder/'
train_dir = os.path.join(base_dir, 'images/train/')
test_dir = os.path.join(base_dir, 'images/test/')
dataset = pd.read_csv(os.path.join(base_dir, 'driver_imgs_list.csv'))
##############
# Groupby
by_drivers = dataset.groupby('subject')
# Groupby unique drivers
unique_drivers = by_drivers.groups.keys() # drivers id
print('There are : ',len(unique_drivers), ' unique drivers')
print('There is a mean of ',round(dataset.groupby('subject').count()['classname'].mean()), ' images by driver.')

activity_map = {'c0': 'Safe driving',
                'c1': 'Texting - right',
                'c2': 'Talking on the phone - right',
                'c3': 'Texting - left',
                'c4': 'Talking on the phone - left',
                'c5': 'Operating the radio',
                'c6': 'Drinking',
                'c7': 'Reaching behind',
                'c8': 'Hair and makeup',
                'c9': 'Talking to passenger'}
NUMBER_CLASSES = 10


def get_cv2_image(path, img_rows, img_cols):
    """
    Function to return an opencv
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_rows, img_cols))  # Reduced size
    return img


def load_train(img_rows, img_cols):
    """
    Return train images and train labels
        """
    train_images = []
    train_labels = []

    # Loop over the training folder
    for classed in tqdm(range(NUMBER_CLASSES)):
        print('Loading directory c{}'.format(classed))
        files = glob(os.path.join('../../../Downloads/distracted_driver_detection_folder/images/train/c' + str(classed), '*.jpg'))
        for file in files:
            img = get_cv2_image(file, img_rows, img_cols)
            train_images.append(img)
            train_labels.append(classed)
    return train_images, train_labels


def read_and_normalize_train_data(img_rows, img_cols):
    """
    Load + categorical + split
    """
    X, labels = load_train(img_rows, img_cols)
    y = to_categorical(labels, 10)  # categorical train label
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # split into train and test
    x_train = np.array(x_train, dtype=np.uint8).reshape(-1, img_rows, img_cols, color_type)
    x_test = np.array(x_test, dtype=np.uint8).reshape(-1, img_rows, img_cols, color_type)

    return x_train, x_test, y_train, y_test


# Loading validation dataset
def load_test(size=200000, img_rows=64, img_cols=64):
    """
    Same as above but for validation dataset
    """
    path = os.path.join('../../../Downloads/distracted_driver_detection_folder/images/test', '*.jpg')
    files = sorted(glob(path))
    X_test, X_test_id = [], []
    total = 0
    files_size = len(files)
    for file in tqdm(files):
        if total >= size or total >= files_size:
            break
        file_base = os.path.basename(file)
        img = get_cv2_image(file, img_rows, img_cols)
        X_test.append(img)
        X_test_id.append(file_base)
        total += 1
    return X_test, X_test_id


def read_and_normalize_sampled_test_data(size, img_rows, img_cols):
    test_data, test_ids = load_test(size, img_rows, img_cols)
    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.reshape(-1, img_rows, img_cols, color_type)
    return test_data, test_ids


img_rows = 64  # dimension of images
img_cols = 64
color_type = 1  # grey
nb_test_samples = 200

# loading train images
x_train, x_test, y_train, y_test = read_and_normalize_train_data(img_rows, img_cols)

# loading validation images
test_files, test_targets = read_and_normalize_sampled_test_data(nb_test_samples, img_rows, img_cols)

# Statistics
# Load the list of names
names = [item[17:19] for item in sorted(glob("../../../Downloads/distracted_driver_detection_folder/images/train/*/"))]
test_files_size = len(np.array(glob(os.path.join('../../../Downloads/distracted_driver_detection_folder/images/test', '*.jpg'))))
x_train_size = len(x_train)
categories_size = len(names)
x_test_size = len(x_test)
print('There are %s total images.\n' % (test_files_size + x_train_size + x_test_size))
print('There are %d training images.' % x_train_size)
print('There are %d total training categories.' % categories_size)
print('There are %d validation images.' % x_test_size)
print('There are %d test images.'% test_files_size)

px.histogram(dataset, x="classname", color="classname", title="Number of images by categories ")

# Find the frequency of images per driver
drivers_id = pd.DataFrame((dataset['subject'].value_counts()).reset_index())
drivers_id.columns = ['driver_id', 'Counts']
px.histogram(drivers_id, x="driver_id",y="Counts" ,color="driver_id", title="Number of images by subjects ")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.figure(figsize = (12, 20))
image_count = 1
BASE_URL = '../../../Downloads/distracted_driver_detection_folder/images/train/'
for directory in os.listdir(BASE_URL):
    if directory[0] != '.':
        for i, file in enumerate(os.listdir(BASE_URL + directory)):
            if i == 1:
                break
            else:
                fig = plt.subplot(5, 2, image_count)
                image_count += 1
                image = mpimg.imread(BASE_URL + directory + '/' + file)
                plt.imshow(image)
                plt.title(activity_map[directory])


##############
batch_size = 40
nb_epoch = 7
models_dir = "saved_models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

checkpointer = ModelCheckpoint(filepath='saved_models/weights_best.hdf5',
                               monitor='val_loss', mode='min',
                               verbose=1, save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)


def create_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    return model


model = create_model()

model.summary()
opt = tf.keras.optimizers.Adam()
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

import timeit
start=timeit.default_timer()
history = model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          epochs=nb_epoch, batch_size=batch_size, verbose=1)
end=timeit.default_timer()
print("time taken : ",end-start)

import matplotlib.pyplot as plt

plt.plot(history.history['val_accuracy'],label='Valdation accuracy')
plt.plot(history.history['accuracy'],label='Training accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'],label='Loss')
plt.legend()
plt.show()

start=timeit.default_timer()
score1 = model.evaluate(x_test, y_test, verbose=1)
stop=timeit.default_timer()
print("time taken : ",stop-start)