import os
from io import BytesIO

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from pathlib import Path

def binarize_array(numpy_array1, threshold=1):
    #numpy_array1 = numpy_array.copy()
    numpy_array1[numpy_array1>=threshold] = 255
    return numpy_array1

def crop(path_im, path_mask, path_out):
  path_mask_out = path_mask.replace("/instance/", "/mask/").replace(".png", ".pbm")

  im = Image.open(path_im)
  mask = Image.open(path_mask).convert("L")

  #plt.imshow(im)
  #plt.show()
  mask = Image.fromarray(binarize_array(np.array(mask)))

  im.putalpha(mask)
  #plt.imshow(im)
  #plt.show()

  rows = np.any(mask, axis=1)
  cols = np.any(mask, axis=0)
  if len(np.where(rows)[0]) > 0:
      ymin, ymax = np.where(rows)[0][[0, -1]]
      xmin, xmax = np.where(cols)[0][[0, -1]]
      rect = int(xmin), int(ymin), int(xmax), int(ymax)
  else:
      rect = -1, -1, -1, -1

  #print(rect)
  im.crop((rect)).save(path_out)
  mask.save(path_mask_out)

PATH = '/home/tsis/projects/automobile/DS/tws_fg_devis/img'

"""
pathlist = Path(PATH).glob('*')
for p in pathlist:
  if os.path.isdir(p):
    dir = str(p).replace("img", "crop")
    print(str(dir))
    os.makedirs(dir)
"""

pathlist = Path(PATH).glob('**/*.jpg')
for path in pathlist:
     # because path is object not string
     path_im = str(path)
     path_mask = path_im.replace("/img/", "/instance/").replace(".jpg", ".png")
     path_out = path_mask.replace("/instance/", "/crop/")
     print(path_im)
     print(path_mask)
     crop(path_im, path_mask, path_out)
