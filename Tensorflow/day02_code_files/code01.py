import numpy as np
import os
import cv2


paths = ['../animal_images/cat/images-1.jpeg', '../animal_images/cat/images-2.jpeg']

imgs = [cv2.imread(path) for path in paths]
# Equivalent to
# imgs = []
# for path in paths:
#     im = cv2.imread(path)
#     imgs.append(im)

img = imgs[0]
# print(img.ndim)
# print(img.shape)
# print(img.dtype)
# print(len(img))
# print(img)
# print(img.tolist())

# cv2.imshow('Test Image', img)
# cv2.waitKey(0) & 0xFF
# cv2.destroyAllWindows()
#
# cv2.imwrite('original.jpg', img)
#
img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC) 		# resize to (200, 200)

# print(img.tolist())
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)	# cv2 load images as BGR, convert it to RGB
# print(img.tolist())
#
# cv2.imwrite('converted.jpg', img)
#
# imgBGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imwrite('reverted.jpg', imgBGR)
#
#
from matplotlib import pyplot as plt

plt.imshow(img)
plt.show()


def plot_images(image):
    # Create figure with 4x4 sub-plots.
    fig, axes = plt.subplots(4, 4)

    for i, ax in enumerate(axes.flat):

        row = i // 4
        col = i % 4

        image_frag = image[row*50:(row+1)*50, col*50:(col+1)*50, :]
        ax.imshow(image_frag)

        # xlabel = '{},{}'.format(row, col)
        # ax.set_xlabel(xlabel)
        # ax.set_xticks([])   # Remove ticks from the plot.
        # ax.set_yticks([])

    plt.show()

# plot_images(img)

img2 = img.reshape(100, 400, 3)
# tmp = img.reshape(3, 40000, 1)
plt.imshow(img2)
plt.show()

flattened_image = img.ravel()

img_f32 = np.float32(img)

normalized_img32 = img_f32/255.

zero_centered_img = (img_f32 - np.mean(img_f32))/np.std(img_f32)

print(img)