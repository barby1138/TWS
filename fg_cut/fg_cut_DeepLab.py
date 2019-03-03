import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

#import imgaug as ia
from skimage.io import imread
#from bilateral_solver import apply_bilateral_files
import scipy.io as sio

import cv2
import scipy
import numpy

import argparse

DO_SHOW = True

class DeepLabModel(object):
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


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

def binarize_array(numpy_array, threshold=1):
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):

            if numpy_array[i][j] >= threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
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
parser.add_argument('--image_url', '-i', help='img url', required=True)
parser.add_argument('--name', '-n', help='FG obj name', required=True)
parser.add_argument('--output', '-o', help='output path', required=True)
args = parser.parse_args()

MODEL_NAME = 'xception_coco_voctrainaug'  # @param ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']

_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {
    'mobilenetv2_coco_voctrainaug':
        'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
    'mobilenetv2_coco_voctrainval':
        'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
    'xception_coco_voctrainaug':
        'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
    'xception_coco_voctrainval':
        'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
}
_TARBALL_NAME = MODEL_NAME + '.tar.gz'

model_dir = '.'
tf.gfile.MakeDirs(model_dir)

download_path = os.path.join(model_dir, _TARBALL_NAME)
print('downloading model, this might take a while...')
if not os.path.isfile(download_path):
  urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)
else:
  print("file exists")
print('download completed! loading DeepLab model...')

MODEL = DeepLabModel(download_path)

print('model loaded successfully!')

#ped
#IMAGE_URL = 'https://bloximages.newyork1.vip.townnews.com/postandcourier.com/content/tncms/assets/v3/editorial/0/b3/0b35580c-9752-11e8-8469-0b011a492a86/5acb91a61f795.image.jpg?resize=893%2C630'
#IMAGE_URL = 'https://i.ytimg.com/vi/Zgdw0XelgnE/maxresdefault.jpg'
#IMAGE_URL = 'https://cdn.vox-cdn.com/thumbor/s7eTfbggoW3GNyfSGk4xVux4HGY=/0x0:1106x705/1200x800/filters:focal(465x238:641x414)/cdn.vox-cdn.com/uploads/chorus_image/image/60486895/Screen_Shot_2018_07_25_at_10.57.00_AM.0.png'
#IMAGE_URL = 'https://img.newatlas.com/segway-drift-w1-eskates-ifa-2018-ride-9.jpg?auto=format%2Ccompress&fit=max&q=60&w=1000&s=a7b84c849bf113649982cfa92f92989c'
#bike
#IMAGE_URL = 'https://www.pe.com/wp-content/uploads/2017/08/0814_nws_rpe-l-traffic-0814.jpg?w=525'

image_url = args.image_url #IMAGE_URL

try:
    f = urllib.request.urlopen(image_url)
    jpeg_str = f.read()
    original_im = Image.open(BytesIO(jpeg_str))

    #print('running deeplab on image %s...' % image_url)

    resized_im, seg_map = MODEL.run(original_im)

    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    save_segmentation(resized_im, seg_image, args.name, args.output)

except IOError:
    print('Cannot retrieve image. Please check url: ' + url)

  
