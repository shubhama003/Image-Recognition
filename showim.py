import matplotlib.pyplot as plt
import cv2
image = cv2.imread("5_test.jpg")
im = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
im = cv2.resize(image, (28, 28))
cv2.imwrite( "grey.jpg", im )

img=cv2.imread("grey.jpg")
plt.imshow(img)
plt.show()
