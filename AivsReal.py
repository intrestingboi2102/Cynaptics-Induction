from keras.utils import to_categorical
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.applications import MobileNetV2, ResNet152, VGG16, EfficientNetB0, InceptionV3
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import tensorflow as tf

def createdataframe(dir):
    image_paths = []
    labels = []
    for label in os.listdir(dir):
        for imagename in os.listdir(os.path.join(dir, label)):
            image_paths.append(os.path.join(dir, label, imagename))
            labels.append(label)
        print(label, "completed")
    return image_paths, labels

def extract_features(images):
    features = []
    for image in tqdm(images):
        img = load_img(image, target_size=(236, 236))
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    features = features.reshape(features.shape[0], 236, 236, 3)  # Reshape all images in one go
    return features

TRAIN_DIR = "Task-1/Data/Train"

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)

train_features = extract_features(train['image'])

x_train = train_features / 255.0

le = LabelEncoder()
le.fit(train['label'])
y_train = le.transform(train['label'])
y_train = to_categorical(y_train, num_classes=2)


model = Sequential()
# Convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(236, 236, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))



# Replace Flatten with GAP
model.add(GlobalAveragePooling2D())

# Fully Connected Layers
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.3))

# Output Layer
model.add(Dense(2, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train, batch_size=20, epochs=18)


def prepareImages(dir):
    image_paths = []
    image_names = []
    for imagename in os.listdir(os.path.join(dir)):
        image_paths.append(os.path.join(dir, imagename))
        image_names.append(imagename)
    return image_paths, image_names

def extract_features(images):
    features = []
    for image in tqdm(images):
        try:
            img = load_img(image, target_size=(236, 236))
            img = np.array(img)
            features.append(img)
        except:
            print("skipping img", image)
    features = np.array(features)
    features = features.reshape(features.shape[0], 236, 236, 3)  # Reshape all images in one go
    return features


TEST_DIR="Task-1/Data/Test"

test = pd.DataFrame()
test["imagepath"], test["Id"] = prepareImages(TEST_DIR)
test_features=extract_features(test["imagepath"])
x_test = test_features/255.0

new_model = model
test["result"] = new_model.predict(x=x_test).argmax(axis=1)
test["Label"] = test["result"].map({1: "Real", 0: "AI"})

result = pd.DataFrame()
result["Id"], result["Label"] = test["Id"] , test["Label"]
result["Id"]= result["Id"].str.replace(".jpg", "", regex=False)
result.to_csv("Task-1/output/result1.csv", index= False)