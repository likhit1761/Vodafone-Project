import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import accuracy_score
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

data = []
labels = []
classes = 43
cur_path = os.getcwd()
for i in range(classes):
    path = os.path.join(cur_path, 'data\Train', str(i))
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '\\' + a)
            image = image.resize((30, 30))  # .convert('LA')
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except Exception as e:
            print(e)
data = np.array(data)
labels = np.array(labels)
np.save('./training/data', data)
np.save('./training/target', labels)
data = np.load('./training/data.npy')
labels = np.load('./training/target.npy')
print(data.shape, labels.shape)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
epochs = 30
history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test))

plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('./training/accuracyPlot.png')

plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('./training/lossPlot.png')

model.save(rf"./training/TSR2.h5")

# model = load_model(rf"./training/TSR1.h5")


# def testing(testcsv):
#     y_test = pd.read_csv(testcsv)
#     label = y_test["ClassId"].values
#     imgs = y_test["Path"].values
#     data = []
#     for img in imgs:
#         image = Image.open(f'data/{img}')
#         image = image.resize((30, 30))
#         data.append(np.array(image))
#     X_test = np.array(data)
#     return X_test, label


# X_test, label = testing('data\Test.csv')
# t, l = testing(r'data/Test.csv')
# print(len(l))
# pred = model.predict(t)
# Y_pred = np.argmax(pred, axis=1)
# print(Y_pred)
# print(accuracy_score(l, Y_pred))
# 12630
# [16  1 38 ... 32  7 10]
# 0.9626286619160729

# model = load_model('./training/TSR1.h5')
# # Classes of trafic signs
# classes = {0: 'Speed limit (20km/h)',
#            1: 'Speed limit (30km/h)',
#            2: 'Speed limit (50km/h)',
#            3: 'Speed limit (60km/h)',
#            4: 'Speed limit (70km/h)',
#            5: 'Speed limit (80km/h)',
#            6: 'End of speed limit (80km/h)',
#            7: 'Speed limit (100km/h)',
#            8: 'Speed limit (120km/h)',
#            9: 'No passing',
#            10: 'No passing veh over 3.5 tons',
#            11: 'Right-of-way at intersection',
#            12: 'Priority road',
#            13: 'Yield',
#            14: 'Stop',
#            15: 'No vehicles',
#            16: 'Veh > 3.5 tons prohibited',
#            17: 'No entry',
#            18: 'General caution',
#            19: 'Dangerous curve left',
#            20: 'Dangerous curve right',
#            21: 'Double curve',
#            22: 'Bumpy road',
#            23: 'Slippery road',
#            24: 'Road narrows on the right',
#            25: 'Road work',
#            26: 'Traffic signals',
#            27: 'Pedestrians',
#            28: 'Children crossing',
#            29: 'Bicycles crossing',
#            30: 'Beware of ice/snow',
#            31: 'Wild animals crossing',
#            32: 'End speed + passing limits',
#            33: 'Turn right ahead',
#            34: 'Turn left ahead',
#            35: 'Ahead only',
#            36: 'Go straight or right',
#            37: 'Go straight or left',
#            38: 'Keep right',
#            39: 'Keep left',
#            40: 'Roundabout mandatory',
#            41: 'End of no passing',
#            42: 'End no passing veh > 3.5 tons'}
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
#
# def test_on_img(img):
#     data = []
#     image = Image.open(img)  # .convert('LA')
#     image = image.resize((30, 30))
#     data.append(np.array(image))
#     X_test = np.array(data)
#     # Y_pred = model.predict_classes(X_test)
#     predict_x = model.predict(X_test)
#     Y_pred = np.argmax(predict_x, axis=1)
#     return image, Y_pred
#
#
# plot, prediction = test_on_img(r'uploads/test2.jpg')
# s = [str(i) for i in prediction]
# a = int("".join(s))
# print("Predicted traffic sign is: ", classes[a])
# plt.imshow(plot)
# plt.show()
