"""
Cuts FGs from VOC DS
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

from collections import namedtuple

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
  

  #return id2label[label].color

def binarize_image(img_path, target_path, threshold):
    image_file = Image.open(img_path)
    image = image_file.convert('L')  # convert image to monochrome
    image = numpy.array(image)
    image = binarize_array(image, threshold)
    imsave(target_path, image)

def binarize_array(numpy_array, threshold=1):
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            #if numpy_array[i][j] == 255: #filter border
             #  numpy_array[i][j] = 0

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

"""
LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])
"""
#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }

#FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
#FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

def unique_array(numpy_array):
    uids = np.zeros((256), dtype=int)
    prev = 0
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
           uids[numpy_array[i][j]] += 1
            
    uids = [idx for idx, uid in enumerate(uids) if uid != 0]

    return uids

def filter_ids(numpy_array, ids):
    uids = np.zeros((256), dtype=int)
    prev = 0
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
          if all(id != numpy_array[i][j] for id in ids) :
             numpy_array[i][j] = 0

    return numpy_array

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help='commands', dest='command')

seg_parser = subparsers.add_parser('seg', help='cut segmentation')
seg_parser.add_argument('--image', '-i', help='DEVIS img filename', required=True)
seg_parser.add_argument('--mask', '-k', help='DEVIS seg mask filename', required=True)
seg_parser.add_argument('--name', '-n', help='FG obj name', required=True)
seg_parser.add_argument('--output', '-o', help='output path', required=True)
seg_parser.add_argument('--filter_ids', '-f', help='filter ids', type=int, nargs='+', required=True)

list_parser = subparsers.add_parser('list', help='list labels')
list_parser.add_argument('--image', '-i', help='DEVIS img filename', required=True)
list_parser.add_argument('--mask', '-k', help='DEVIS seg mask filename', required=True)

args = parser.parse_args()

im = Image.open(args.image)
seg_map = Image.open(args.mask).convert('L')
seg_im = label_to_color_image(numpy.array(seg_map)).astype(np.uint8)

#print(numpy.array(seg_im))
if args.command == "seg":

  print(unique_array(numpy.array(seg_map)))

  seg_map_f = filter_ids(numpy.array(seg_map), args.filter_ids)
  print(unique_array(seg_map_f))

  seg_im_filtered = label_to_color_image(seg_map_f).astype(np.uint8)

  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 2, width_ratios=[6, 6])
    
  plt.subplot(grid_spec[0])
  plt.imshow(seg_im)
  plt.axis('off')
  plt.title('src image')

  plt.subplot(grid_spec[1])
  plt.imshow(seg_im_filtered)
  plt.axis('off')
  plt.title('filtered image')

  if DO_SHOW:
    plt.show()

  seg_im = seg_im_filtered

  save_segmentation(im, numpy.array(seg_im), args.name, args.output)
if args.command == "list":
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 2, width_ratios=[6, 6])
  
  plt.subplot(grid_spec[0])
  plt.imshow(im)
  plt.axis('off')
  plt.title('src image')

  plt.subplot(grid_spec[1])
  plt.imshow(seg_im)
  plt.axis('off')
  plt.title('seg image')

  if DO_SHOW:
    plt.show()

  uids = unique_array(numpy.array(seg_map))
  for i in uids:
    print("name: {} id: {}".format(id2label[i].name, id2label[i].id))
