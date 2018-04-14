import cv2
from matplotlib import pyplot as plt
from keras import backend as K
from keras.models import load_model
model = load_model('model.h5')


image = cv2.imread("5_test.jpg")
img_rows, img_cols = 28, 28

x_test = cv2.resize(image, (28, 28)) 

x_test = cv2.cvtColor(x_test,cv2.COLOR_RGB2GRAY)


if K.image_data_format() == 'channels_first':
    print('reshape1')
    x_test = x_test.reshape(1, 1, img_rows, img_cols)
else:
    print('reshape2')
    x_test = x_test.reshape(1, img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255
y_test = [5]
score = model.evaluate(x_test, y_test, verbose=0)

print('donee')
print('score of image 5 = ')
print(score[1])

#plt.imshow(x_test)
#plt.show()
