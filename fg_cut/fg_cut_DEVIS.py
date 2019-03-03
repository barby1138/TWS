"""
Cuts FGs from DEVIS DS
https://davischallenge.org/
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

DO_SHOW = False

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

def save_segmentation(image, seg_image, obj_name, out_path):
  image.save(out_path + '/' + obj_name + '.jpg')

  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 6])

  immask_rgb = Image.fromarray(seg_image) #.convert('RGB')
  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.imshow(immask_rgb, alpha=0.4)
  plt.axis('off')
  plt.title('src image')
  #immask_rgb.save('./images/seg_img.jpg')

  immask = Image.fromarray(seg_image).convert('RGB').convert('L')
  immask1 = Image.fromarray(binarize_array(numpy.array(immask)))
  immask1_inv = Image.fromarray(inv_array(binarize_array(numpy.array(immask))))
  plt.subplot(grid_spec[1])
  plt.imshow(immask1_inv)
  plt.axis('off')
  plt.title('inv mask')
  immask1_inv.save(out_path + '/' + obj_name + '.pbm')

  image.putalpha(immask1)
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
parser.add_argument('--mask', '-k', help='DEVIS seg mask filename', required=True)
parser.add_argument('--name', '-n', help='FG obj name', required=True)
parser.add_argument('--output', '-o', help='output path', required=True)
args = parser.parse_args()

im = Image.open(args.image)
seg_im = Image.open(args.mask)
    
save_segmentation(im, numpy.array(seg_im), args.name, args.output)


