import os
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from keras import models
from keras import layers
from keras.optimizers import Adam
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout
from keras.applications.inception_v3 import InceptionV3
from keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def fine_tune_inception_v3():
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

    model = models.Sequential()
    model.add(base_model)  # add pre_trained layers
    model.add(GlobalAveragePooling2D())
    # model.add(Flatten()) # flatten to 1-D vector to prepare for fully connected layers
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(20, activation='softmax'))

    # Freeze pre-trained layers
    print('Number of trainable weights before freezing the base layer:', len(model.trainable_weights))
    model.layers[0].trainable = False
    print('Number of trainable weights after freezing the base layer:', len(model.trainable_weights))

    model.compile(tf.keras.optimizers.Adam(learning_rate=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    return model

def preprocess(img_path):
    # Thay đổi kích thước tất cả ảnh về 299x299
    img = image.load_img(img_path, target_size=(299, 299))
    # Chuyển ảnh thành array
    img = image.img_to_array(img)
    # Tiền xử lý cho mô hình incenption v3
    preprocessed_img = preprocess_input(img)

    return np.expand_dims(preprocessed_img, axis=0)

def display_4_images(images, titles):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    for i, (image, title) in enumerate(zip(images, titles)):
        axs[i // 2, i % 2].imshow(image)
        axs[i // 2, i % 2].set_title(title)
        axs[i // 2, i % 2].axis('off')
    plt.show()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_path", type=str, default='data/golden-retriever-dog-breed-info-2.jpeg',
                    help="Path to the image to be used")
    ap.add_argument("--folder_path", type=str, default='data2/',
                    help="Path to the folder to be used")
    ap.add_argument("--model_path", type=str, default='model/dog_breed_inceptionv3.keras',
                    help="Path to the Model to be used")
    ap.add_argument("--threshold", type=int, default=0.9,
                    help="Set threshold")
    args = vars(ap.parse_args())

    # Đầu vào
    img_path = args["img_path"]
    folder_path = args["folder_path"]
    model_path = args["model_path"]
    threshold = args["threshold"]

    # Nạp mô hình
    # model = tf.keras.models.load_model(model_path)
    model = fine_tune_inception_v3()

    model.load_weights('./checkpoints/my_checkpoint').expect_partial()

    # Show the model architecture
    model.summary()

    # Read the labels.csv file and check shape and records
    labels_all = pd.read_csv('labels.csv')
    # Loading number or each breed
    breed_all = labels_all['breed']
    breed_count = breed_all.value_counts()
    # Selecting all breeds because i have high computation power
    CLASS_NAME = breed_all.unique()[:40]

    print('Những loài có thể dự đoán:')
    for label in CLASS_NAME:
        print(label)
    print('_________________________________________________________________')

    # Duyệt qua các thư mục con trong thư mục data
    for folder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, folder)

        # Lấy tên giống chó từ tên thư mục
        breed = os.path.basename(subfolder_path)

        # Lấy danh sách các ảnh trong thư mục
        image_paths = [
            os.path.join(subfolder_path, image)
            for image in os.listdir(subfolder_path)
        ]

        # Dự đoán và hiển thị 4 ảnh đầu tiên
        images = []
        titles = []
        for path in image_paths:
            origin_img = image.load_img(path)
            preprocessed_img = preprocess(path)
            y_pred = model.predict(preprocessed_img)
            name = CLASS_NAME[np.argmax(y_pred)]
            conf = np.max(y_pred)
            if conf >= threshold:
                title = f"Predicted: {name}; Confidence: {round(float(conf), 2)}"
            else:
                title = "Unknown"
            images.append(origin_img)
            titles.append(title)


        def display_multiple_images(images, titles):
            # Tính toán số hàng và cột cần thiết để hiển thị tất cả ảnh
            num_rows = len(images) // 2 + len(images) % 2  # Thêm một hàng nếu số ảnh lẻ
            num_cols = min(len(images), 2)  # Số cột tối đa là 2

            # Tạo ma trận subplot với số hàng và cột tính toán được
            fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 10))

            for i, (image, title) in enumerate(zip(images, titles)):
                if num_rows > 1:
                    axs[i // num_cols, i % num_cols].imshow(image)
                    axs[i // num_cols, i % num_cols].set_title(title)
                    axs[i // num_cols, i % num_cols].axis('off')
                else:
                    axs[i % num_cols].imshow(image)
                    axs[i % num_cols].set_title(title)
                    axs[i % num_cols].axis('off')

            plt.show()
        display_multiple_images(images, titles)
