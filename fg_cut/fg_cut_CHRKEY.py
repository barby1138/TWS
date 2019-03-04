"""
Cuts FGs from chroma key
"""

import os
from io import BytesIO
import tarfile
import tempfile

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import cv2
import scipy
import numpy

import argparse

DO_SHOW = True

def create_pascal_label_colormap():

  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):

  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

def binarize_image(img_path, target_path, threshold):
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = numpy.array(image)
    image = binarize_array(image, threshold)
    imsave(target_path, image)

def binarize_array(numpy_array, threshold=1):
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):

 #           if numpy_array[i][j] >= 110 and numpy_array[i][j] <= 124:
  #              numpy_array[i][j] = 0

            if numpy_array[i][j] >= threshold:
                numpy_array[i][j] = 255 #0
            else:
                numpy_array[i][j] = 0 #255
    return numpy_array

def inv_array(numpy_array, threshold=1):
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            numpy_array[i][j] = 255 - numpy_array[i][j]
    
    return numpy_array

def remove_green(img):
    """
    Docstring:
        Remove green-ish background from image given a threshold.

    Parameters
    ----------
    img : numpy.array containing 4 channel image (RGBa).
    """
    
    norm_factor = 255

    """
    Obtain the ratio of the green/red/blue
    channels based on the max-bright of 
    the pixel.
    """
    
    red_ratio = img[:, :, 0] / norm_factor
    green_ratio = img[:, :, 1] / norm_factor
    blue_ratio = img[:, :, 2] / norm_factor

    """
    Darker pixels would be around 0.
    In order to ommit removing dark pixels we
    sum .3 to make small negative numbers to be
    above 0.
    """
    
    red_vs_green = (red_ratio - green_ratio) + .3
    blue_vs_green = (blue_ratio - green_ratio) + .3

    """
    Now pixels below 0. value would have a
    high probability to be background green
    pixels.
    """
    red_vs_green[red_vs_green < 0] = 0
    blue_vs_green[blue_vs_green < 0] = 0

    """
    Combine the red(blue) vs green ratios to
    set an alpha layer with valid alpha-values.
    """
    alpha = (red_vs_green + blue_vs_green) * 255
    alpha[alpha > 99] = 255

    """
    Set the alpha layer
    """
    img[:, :, 3] = alpha
    
    return img

def save_segmentation(image, seg_image, obj_name, out_path):
  image.save(out_path + '/' + obj_name + '.jpg')
  
  img = seg_image
  img_rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
  fgmask = remove_green(img_rgba)          
  
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 6])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('src image')

  immask1 = Image.fromarray(fgmask[:,:,3])
  immask1_inv = Image.fromarray(inv_array(binarize_array(numpy.array(immask1))))
  plt.subplot(grid_spec[1])
  plt.imshow(immask1_inv)
  plt.axis('off')
  plt.title('inv mask')
  immask1_inv.save(out_path + '/' + obj_name + '.pbm')

  image = Image.fromarray(fgmask)
  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.axis('off')
  plt.title('cut FG')
  image.save(out_path + '/' + obj_name + '_a.png')

  if DO_SHOW:
    plt.show()

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

parser = argparse.ArgumentParser()
parser.add_argument('--image', '-i', help='DEVIS img filename', required=True)
parser.add_argument('--name', '-n', help='FG obj name', required=True)
parser.add_argument('--output', '-o', help='output path', required=True)
args = parser.parse_args()

im = Image.open(args.image)

save_segmentation(im, numpy.array(im), args.name, args.output)
