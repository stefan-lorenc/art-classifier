import urllib.request
from tqdm import tqdm
from selenium_base import driver_creation
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import re
import splitfolders
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

img_width, img_height = 224, 224


def classify(img_path, model, category):
    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    img_batch = np.expand_dims(input_arr, axis=0)

    prediction = model.predict(img_batch)

    if prediction[0].tolist()[0] == 1:
        print(category[0])
    else:
        print(category[1])


def painting_collect(link, directory):
    driver = driver_creation(is_headless=True)
    driver.get(link)

    paintings = driver.find_elements(By.CLASS_NAME, 'image')

    paintings = [monet.find_element(By.TAG_NAME, 'img').get_attribute('src') for monet in paintings]

    count = 0

    for monet in tqdm(paintings):
        monet = re.sub('/\d+px', '/500px', monet)
        driver.get(monet)
        driver.find_element(By.XPATH, '/html/body/img').screenshot(f'{directory}/painting_{count}.png')
        count += 1


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def get_model(input_shape=None, train_generator=None, validation_generator=None, epochs=None, weights=None):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc', f1_m])

    ss_train = train_generator.n // train_generator.batch_size
    ss_val = validation_generator.n // validation_generator.batch_size


    model.fit(
        train_generator,
        steps_per_epoch=ss_train,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=ss_val,
        class_weight=weights)



    return model


def train():

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    class_golden = r'data/golden_age'
    class_monet = r'data/monet'

    sup_w = len([entry for entry in os.listdir(class_monet) if os.path.isfile(os.path.join(class_monet, entry))]) / len(
        [entry for entry in os.listdir(class_golden) if os.path.isfile(os.path.join(class_golden, entry))])

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    batch_size = 16

    train_generator = train_datagen.flow_from_directory(
        'data_structured/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        'data_structured/val',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

    model = get_model(input_shape=input_shape, train_generator=train_generator,
                      validation_generator=validation_generator, epochs=5, weights={0: 1, 1: sup_w})

    model.save('curator.h5')


if __name__ == '__main__':
    # painting_collect()

    # splitfolders.ratio('data', output="data_structured", seed=11, ratio=(0.8, 0.1, 0.1))

    train()

    model = keras.models.load_model('curator.h5', custom_objects={"f1_m": f1_m})

    classify('data_structured/test/golden_age/painting_614.png', model, ['golden_age', 'monet'])
    classify('data_structured/test/monet/painting_614.png', model, ['golden_age', 'monet'])

