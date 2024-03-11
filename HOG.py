import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure
from skimage import feature

#đọc ảnh
img = cv2.imread('./download.jpg')
cv2.imshow('image',img)

#resize ảnh gốc
up_width = 496
up_height = 496
up_points = (up_width, up_height)
resize_up = cv2.resize(img, up_points, interpolation= cv2.INTER_LINEAR)

#tạo ảnh xám từ ảnh vừa resize
gray =cv2.cvtColor(resize_up, cv2.COLOR_BGR2GRAY)
cv2.imshow('grey image',gray)

print('Kích thước ảnh gốc: ', img.shape)
print('Kích thước ảnh resize và ảnh xám : ', gray.shape)

#trích xuất đặc trưng HOG
(H, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), transform_sqrt=True, block_norm="L2",
    visualize=True)

hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
print('Kích thước ảnh hog: ', hogImage.shape)

cv2.imshow('HOG Image',hogImage)
cv2.waitKey(0)