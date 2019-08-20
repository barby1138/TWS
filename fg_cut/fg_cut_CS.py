"""
Cuts FGs from CS DS
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

import os, sys

from pathlib import Path

import json

DO_SHOW = True
MAKE_INV_MASK = False

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

def binarize_array(numpy_array1, threshold=1):
    #numpy_array1 = numpy_array.copy()
    numpy_array1[numpy_array1>=threshold] = 255
    return numpy_array1

def inv_array(numpy_array, threshold=1):
    numpy_array = 255 - numpy_array
    return numpy_array

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

#np.unique?
def unique_array(numpy_array):
    uids = np.zeros((256), dtype=int)
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
           uids[numpy_array[i][j]] += 1
            
    uids = [idx for idx, uid in enumerate(uids) if uid != 0]

    return uids

def filter_ids(numpy_array, ids):
    numpy_array1 = numpy_array.copy()

    numpy_array1[np.isin(numpy_array1, ids, invert=True)] = 0
    return numpy_array1

class Instance(object):
    instID     = 0
    labelID    = 0
    pixelCount = 0
    medDist    = -1
    distConf   = 0.0

    def __init__(self, imgNp, instID):
        if (instID == -1):
            return
        self.instID     = int(instID)
        self.labelID    = int(self.getLabelID(instID))
        self.pixelCount = int(self.getInstancePixels(imgNp, instID))

    def getLabelID(self, instID):
        if (instID < 1000):
            return instID
        else:
            return int(instID / 1000)

    def getInstancePixels(self, imgNp, instLabel):
        return (imgNp == instLabel).sum()

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def toDict(self):
        buildDict = {}
        buildDict["instID"]     = self.instID
        buildDict["labelID"]    = self.labelID
        buildDict["pixelCount"] = self.pixelCount
        buildDict["medDist"]    = self.medDist
        buildDict["distConf"]   = self.distConf
        return buildDict

    def fromJSON(self, data):
        self.instID     = int(data["instID"])
        self.labelID    = int(data["labelID"])
        self.pixelCount = int(data["pixelCount"])
        if ("medDist" in data):
            self.medDist    = float(data["medDist"])
            self.distConf   = float(data["distConf"])

    def __str__(self):
        return "("+str(self.instID)+")"
        

#
# review
#

class Cutter(object):
  
  def __init__(self, path):
      self.pathlist = list(Path(path).glob('*.png'))
      self.im_path = str(self.pathlist[0])
      self.mask_path = self.im_path.replace("leftImg8bit", "gtFine").replace(".png", "_instanceIds.png")
      #print(self.im_path)
      #print(self.mask_path)
      self.idx = 0
      self.instanceId_1 = 0

      self.im = Image.open(self.im_path)
      self.seg_map = Image.open(self.mask_path).convert('L')
      self.seg_im = label_to_color_image(numpy.array(self.seg_map)).astype(np.uint8)
      self.img = Image.open(self.mask_path)
      # Image as numpy array
      self.imgNp = np.array(self.img)
      
      plt.figure(figsize=(15, 5))
      self.grid_spec = gridspec.GridSpec(1, 2, width_ratios=[6, 6])
  
      plt.connect('key_press_event', self.key)
      plt.connect('button_press_event', self.button)
  
      self.render_pic()
      plt.show()

  def render_pic(self):

      #self.grid_spec = gridspec.GridSpec(1, 2, width_ratios=[6, 6])
  
      plt.subplot(self.grid_spec[0])
      plt.imshow(self.im)
      plt.title('src image')

      plt.subplot(self.grid_spec[1])
      plt.imshow(self.seg_im)
      plt.title('seg image')
     
      plt.draw()

  def init_pic(self, inc):
      self.idx += inc

      print(self.idx)
      self.im_path = str(self.pathlist[self.idx])
      #print(self.im_path)
      self.mask_path = self.im_path.replace("leftImg8bit", "gtFine").replace(".png", "_instanceIds.png")
      #print(self.im_path)
      #print(self.mask_path)

      self.im = Image.open(self.im_path)
      self.seg_map = Image.open(self.mask_path).convert('L')
      self.seg_im = label_to_color_image(numpy.array(self.seg_map)).astype(np.uint8)
      self.img = Image.open(self.mask_path)

      self.imgNp = np.array(self.img)
  
      self.render_pic()

  def render_inst(self, point):
      print(point[0])
      self.instanceId_1 = self.imgNp[point[0][1]][point[0][0]]
      self.imgNp1 = binarize_array(filter_ids(self.imgNp, [self.instanceId_1]))
      self.imgNp1_im = Image.fromarray(self.imgNp1).convert('RGB').convert('L')
      self.imgNp1_im.save('.' + '/inst_' + str(self.idx) + '_' + str(self.instanceId_1) + '.pbm')

      id = self.instanceId_1 // 1000
      print(id2label[id].name)

      data = {}
      data["class_name"] = id2label[id].name
      with open('.' + '/inst_' + str(self.idx) + '_' + str(self.instanceId_1) + '.json', 'w') as outfile:  
        json.dump(data, outfile)

      self.crop()

      plt.subplot(self.grid_spec[1])
      #plt.imshow(Image.fromarray(imgNp1))
      plt.imshow(self.imgNp1_im)
      plt.imshow(self.im, alpha=0.4)
      plt.draw()

  def crop(self):
      #path_mask_out = path_mask.replace("/instance/", "/mask/").replace(".png", ".pbm")

      im = self.im
      try:
        im.save('.' + '/inst_' + str(self.idx) + '_' + str(self.instanceId_1) + '.jpg')
      except OSError (err):
        print("save jpg error %s" % err)
    
      im_crop = im.copy()
      im_crop.putalpha(self.imgNp1_im)
    
      rows = np.any(self.imgNp1_im, axis=1)
      cols = np.any(self.imgNp1_im, axis=0)
      if len(np.where(rows)[0]) > 0:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        rect = int(xmin), int(ymin), int(xmax), int(ymax)
      else:
        rect = -1, -1, -1, -1

      print(rect)

      im_crop.crop((rect)).save('.' + '/inst_' + str(self.idx) + '_' + str(self.instanceId_1) + '.png')
    
  def button(self, event):
        x, y = int(event.xdata), int(event.ydata)
        print((x, y))
        point = [[x,y]]
        self.render_inst(point)

  def key(self, event):
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'x':
        print("close")
        plt.close()

    if event.key == 'd':
        print("dump")
        print(instanceId_1)

    if event.key == 'right':
        self.init_pic(1)

    if event.key == 'left':
        self.init_pic(-1)

parser = argparse.ArgumentParser()
parser.add_argument('--image', '-i', help='DEVIS img folder', required=True)

args = parser.parse_args()

cutter = Cutter(args.image)

        
  
