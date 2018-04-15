
# coding: utf-8

# In[1]:


pwd


# In[10]:



from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import pickle
import h5py
import numpy as np
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

model=load_model('my_model.h5')


# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print('initial shape')
# print('x_test ',x_test.shape)
# print('y_test ',y_test.shape)


# # print(x_test);
# # print(y_test.shape)

# if K.image_data_format() == 'channels_first':
#     print('reshape1')
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     print('reshape2')
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print('x_test shape: ' , x_test.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# print('x_test final : ')
# print(x_test)

# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))

# #with open("model.pkl","rb") as input:
# 	#model = pickle.load(input)

# #with open("model.pkl","wb") as output :
#   	#pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
       
# model.save('my_model.h5')

# #image = cv2.imread("5_test.jpg")
# #resized_image = cv2.resize(image, (28, 28)) 
# #image = resized_image.reshape(1,1,28,28)

# #x_test = image
# #y_test = [5]

# score = model.evaluate(x_test, y_test, verbose=0)


# # print(x_test.shape)
# # print('\n')
# # print(y_test.shape)


# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

image = cv2.imread("2.jpg")
img_rows, img_cols = 28, 28

x_test1 = cv2.resize(image, (28, 28)) 

x_test1 = cv2.cvtColor(x_test1,cv2.COLOR_RGB2GRAY)
# cv2.imwrite("grey1.jpg",x_test1)
# x_test2=cv2.imread("grey1.jpg")

# print(x_test2.shape)

# b ,g ,r = cv2.split(x_test2);

# x_test1=cv2.resize(b,(28,28));

print(x_test1.shape)

if K.image_data_format() == 'channels_first':
    print('reshape1')
    x_test1 = x_test1.reshape(1, 1, img_rows, img_cols)
else:
    print('reshape2')
    x_test1 = x_test1.reshape(1, img_rows, img_cols, 1)

x_test1 = x_test1.astype('float32')
x_test1 /= 255
y_test1 = np.array([[ 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])
# y_test1=np.array([[5,5,5,5,5,5,5,5,5,5]])

print(x_test1.shape)

# print('\n')
# print("y_test is ")

# print(y_test1.shape)

score = model.evaluate(x_test1, y_test1, verbose=0)

print('done')
print('score of image 5 = ')
print(score[1])
print(score[0])
score=model.predict_classes(x_test1)
print(score)
# x_test1=cv2.resize(x_test1,(28,28))

