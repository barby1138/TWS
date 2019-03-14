import warnings; warnings.filterwarnings('ignore');

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.misc import imread

import skimage
from skimage import img_as_ubyte
from skimage.transform import rescale
from skimage.transform import rotate
from skimage import io
from skimage import filters
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank

from PIL import Image
import cv2

#File where to save the final image of this document
output_filename = 'outfile.png'

DO_SHOW = False

"""
Load images
"""
bg = imread('./img_bg/bg.png', mode='RGBA')
girl_0 = imread('./img_fg/girl_a.png', mode='RGBA')
girl_0_mask = imread('./img_fg/girl.pbm')

cc = imread('./img_fg/classic_car_a.png', mode='RGBA')
stormtrooper = imread('./img_fg/birds_a.png', mode='RGBA')
lucia = imread('./img_fg/lucia_a.png', mode='RGBA')
falcon = imread('./img_fg/zebra_a.png', mode='RGBA')
#girl_1 = imread('./img_fg/girl_a_1.png', mode='RGBA')
girl_1 = imread('./img_fg/girl_a_0.png', mode='RGBA')

"""
Display
"""
fig, ax = plt.subplots(1, 7)

ax[0].imshow(bg)
ax[1].imshow(girl_0)
ax[2].imshow(cc)
ax[3].imshow(stormtrooper)
ax[4].imshow(lucia)
ax[5].imshow(falcon)
ax[6].imshow(girl_1)

fig.set_size_inches(18, 5)
if DO_SHOW:
    plt.show()

def blend(bg, img, coord=(0, 0), scale=None, angle=None):
    #Perform scaling
    if not(scale is None):
        img = rescale(
            img, 
            scale)
    
    #Perform rotation
    if not(angle is None):
        img = rotate(
            img, 
            angle,
            resize=True, 
            mode='edge')
    
    img = img_as_ubyte(img)
    
    (x_size, y_size, _) = img.shape

    (y_ini, x_ini) = coord
    x_end = x_ini + x_size
    y_end = y_ini + y_size

    bg_crop = bg[
        x_ini:x_end,
        y_ini:y_end, 
        :]

    pixel_preserve = (img[:, :, -1] > 10)
    bg_crop[pixel_preserve] = img[pixel_preserve]

    bg[x_ini:x_end, y_ini:y_end, :] = bg_crop

    return bg

def PIL2array1C(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])

def PIL2array3C(img):
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

def binarize_array(numpy_array, threshold=1):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] < 255: #filter cars
                numpy_array[i][j] = 0

            if numpy_array[i][j] >= threshold:
                numpy_array[i][j] = 255 #0
            else:
                numpy_array[i][j] = 0 #255
    return numpy_array

img = bg.copy()

img = blend(
    img, falcon,
    coord=(1000, 700),
    scale=.5,
    angle=-10)
print("1")

img = blend(
    img, girl_0,
    coord=(100, 800),
    scale=.2)
print("2")

img = blend(
    img, cc,
    coord=(300, 1000),
    angle=5,
    scale=.2)
print("3")
img = blend(
    img, lucia,
    coord=(1900, 750),
    scale=.2)
print("4")
img = blend(
    img, stormtrooper,
    coord=(50, 60),
    scale=.1)
print("5")
img = blend(
    img, stormtrooper,
    coord=(115, 130),
    angle=20,
    scale=.4)
print("6")
img = blend(
    img, stormtrooper,
    coord=(250, 50),
    angle=15,
    scale=.6)
print("7")
img = blend(
    img,  girl_1,
    coord=(400, 1000),
    angle=-10,
    scale=.6)
print("8")
fig, ax = plt.subplots(1, 1)
ax.imshow(img)
fig.set_size_inches(10, 10)
if DO_SHOW:
    plt.show()
Image.fromarray(img).save('./img_comc.png')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def channel_adjust(img, curve, channel):
    regular_inntervals = np.linspace(0, 1, len(curve))
    
    adjustment_flat = np.interp(
        img[:, :, channel].ravel(), 
        regular_inntervals,
        curve)
    shape = img[:, :, channel].shape
    
    adjustment = adjustment_flat.reshape(shape)
    img[:, :, channel] = adjustment
    
    return

#Make adjust curves
red_adjust_curve = sigmoid(np.arange(-3, 5, .35))
green_adjust_curve = sigmoid(np.arange(-5, 5, .45))
blue_adjust_curve = sigmoid(np.arange(-2.3, 2.3, .2))

fig, ax = plt.subplots(1, 3)

img_effect = skimage.img_as_float(
    img.copy())

channel_adjust(
    img_effect, 
    red_adjust_curve, 
    channel=0)

channel_adjust(
    img_effect, 
    green_adjust_curve, 
    channel=1)

channel_adjust(
    img_effect, 
    blue_adjust_curve, 
    channel=2)

ax[0].imshow(img)
ax[0].set_title('Original image')
ax[1].imshow(img_effect)
ax[1].set_title('Image after applyed curve adjustment')

pd.Series(red_adjust_curve).plot(
    c='r', ax=ax[2], title='Adjustment curves')
pd.Series(green_adjust_curve).plot(
    c='g', ax=ax[2])
pd.Series(blue_adjust_curve).plot(
    c='b', ax=ax[2])

fig.set_size_inches(14, 7)

fig.set_size_inches(20, 5)
fig.set_tight_layout('tight')
if DO_SHOW:
    plt.show()

#Image.fromarray(img_effect).save('./img_comp_1.png')

blurred = filters.gaussian(
    img_effect.copy(),
    sigma=3,
    multichannel=True)

shape_0 = blurred.shape[0]
shape_1 = blurred.shape[1]
blur_gradient = (1 - np.arange(0, 1, 1 / shape_1))

blur_gradient = blur_gradient.reshape(
    1, -1
).repeat(
    shape_0, axis=0
).reshape(shape_0, shape_1)

blurred[:, :, 3] = 1 - blur_gradient

fig, ax = plt.subplots(1, 1)
plt.imshow(blurred)
fig.set_size_inches(12, 12)

if DO_SHOW:
    plt.show()

def merge_alpha(img, bg):

    src_rgb = img[..., :3]
    src_a = img[..., 3]

    dst_rgb = bg[..., :3]
    dst_a = bg[..., 3]

    out_a = src_a + dst_a * (1.0 - src_a)
    out_rgb = (
        src_rgb*src_a[..., None] + dst_rgb*dst_a[..., None]*(1.0 - src_a[..., None])) / out_a[..., None]

    out = np.zeros_like(bg)
    out[..., :3] = out_rgb
    out[..., 3] = out_a
    
    return out

final = merge_alpha(blurred, img_effect)

fig, ax = plt.subplots(1, 1)

plt.imshow(final)

fig.set_size_inches(12, 12)

if DO_SHOW:
    plt.show()

io.imsave(
    output_filename, 
    skimage.img_as_uint(final))
