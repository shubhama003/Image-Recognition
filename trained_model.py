from keras.models import load_model
from keras.datasets import mnist
from PIL import Image
import matplotlib.pyplot as plt
import cv2
model = load_model('model.h5')

img = Image.open('5_test.jpg').convert('LA')
img.save('greyscale.png')

image = cv2.imread("greyscale.png")
im = cv2.resize(image, (28, 28)) 

im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
plt.imshow(im)
plt.show()

pr = model.predict_classes(im.reshape(1,28,28,1))
print(pr)


