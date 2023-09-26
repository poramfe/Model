import cv2
import numpy as np

vertical = cv2.imread(filename="dataset/images/vertical_100x1.png", flags=cv2.IMREAD_GRAYSCALE)
horizontal = cv2.imread(filename="dataset/images/horizontal_100x1.png", flags=cv2.IMREAD_GRAYSCALE)

print(f"vertical.shape : {vertical.shape}")
print(f"horizontal.shape : {horizontal.shape}")

########## 1) Image preparation
vertical = vertical / 255
horizontal = horizontal / 255

vertical_flattened = vertical.flatten()
# print(vertical_flattened)
horizontal_flattened = horizontal.flatten()
# print(horizontal_flatten)

########## 2) Create Image recognition classifier
horizontal_sum = np.sum(horizontal_flattened)
vertical_sum = np.sum(vertical)
print(f"horizontal sum : {horizontal_sum}")
print(f"vertical sum : {vertical_sum}")
print(f"vertical : {vertical_flattened}")
print(f"horizontal : {horizontal_flattened}")

# cv2.imshow(winname="Vertical image", mat=cv2.resize(src=vertical, dsize=(500, 500), interpolation=0))   # interpolation : 보간법
# cv2.imshow(winname="Horizontal image", mat=cv2.resize(src=horizontal, dsize=(500, 500), interpolation=0))

# cv2.imshow(winname="Vertical Flattened image", mat=cv2.resize(src=vertical_flattened, dsize=(50,450), interpolation=0))
# cv2.imshow(winname=" Horizontal Flattened image", mat=cv2.resize(src=horizontal_flattened, dsize=(450, 50), interpolation=0))
# cv2.waitKey(delay=0)


print("end")