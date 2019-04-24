#import argparse
import glob
import sys
import os
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom
import cv2
import numpy as np
#import random
from PIL import Image, ImageOps
#import scipy
from multiprocessing import Pool
from functools import partial
import signal
#import time


POISSON_BLENDING_DIR = './EXP'
INVERTED_MASK = False
NUMBER_OF_WORKERS = 4
BLENDING_LIST = ['gaussian','poisson', 'none', 'box', 'motion']

#from defaults import *
sys.path.insert(0, POISSON_BLENDING_DIR)
from pb import *
import math
from pyblur.pyblur import *
from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def randomAngle(kerneldim):
    """Returns a random angle used to produce motion blurring

    Args:
        kerneldim (int): size of the kernel used in motion blurring

    Returns:
        int: Random angle
    """ 
    kernelCenter = int(math.floor(kerneldim/2))
    numDistinctLines = kernelCenter * 4
    validLineAngles = np.linspace(0,180, numDistinctLines, endpoint = False)
    angleIdx = np.random.randint(0, len(validLineAngles))
    return int(validLineAngles[angleIdx])

def LinearMotionBlur3C(img):
    """Performs motion blur on an image with 3 channels. Used to simulate 
       blurring caused due to motion of camera.

    Args:
        img(NumPy Array): Input image with 3 channels

    Returns:
        Image: Blurred image by applying a motion blur with random parameters
    """
    lineLengths = [3,5,7,9]
    lineTypes = ["right", "left", "full"]
    lineLengthIdx = np.random.randint(0, len(lineLengths))
    lineTypeIdx = np.random.randint(0, len(lineTypes)) 
    lineLength = lineLengths[lineLengthIdx]
    lineType = lineTypes[lineTypeIdx]
    lineAngle = randomAngle(lineLength)
    blurred_img = img
    for i in range(3):
        blurred_img[:,:,i] = PIL2array1C(LinearMotionBlur(img[:,:,i], lineLength, lineAngle, lineType))
    blurred_img = Image.fromarray(blurred_img, 'RGB')
    return blurred_img

def overlap(a, b):
    '''Find if two bounding boxes are overlapping or not. This is determined by maximum allowed 
       IOU between bounding boxes. If IOU is less than the max allowed IOU then bounding boxes 
       don't overlap

    Args:
        a(Rectangle): Bounding box 1
        b(Rectangle): Bounding box 2
    Returns:
        bool: True if boxes overlap else False
    '''
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    
    if (dx>=0) and (dy>=0) and float(dx*dy) > MAX_ALLOWED_IOU*(a.xmax-a.xmin)*(a.ymax-a.ymin):
        return True
    else:
        return False

def get_list_of_images(root_dir, N=1):
    '''Gets the list of images of objects in the root directory. The expected format 
       is root_dir/<object>/<image>.jpg. Adds an image as many times you want it to 
       appear in dataset.

    Args:
        root_dir(string): Directory where images of objects are present
        N(int): Number of times an image would appear in dataset. Each image should have
                different data augmentation
    Returns:
        list: List of images(with paths) that will be put in the dataset
    '''
    img_list = glob.glob(os.path.join(root_dir, '**/*.jpg'),recursive=True)
    print(len(img_list))
    img_list_f = []
    for i in range(N):
        img_list_f = img_list_f + random.sample(img_list, len(img_list))
    return img_list_f

def get_mask_file(img_file):
    '''Takes an image file name and returns the corresponding mask file. The mask represents
       pixels that belong to the object. Default implentation assumes mask file has same path 
       as image file with different extension only. Write custom code for getting mask file here
       if this is not the case.

    Args:
        img_file(string): Image name
    Returns:
        string: Correpsonding mask file path
    '''
    mask_file = img_file.replace('.jpg','.pbm')
    return mask_file

def get_labels(imgs):
    '''Get list of labels/object names. Assumes the images in the root directory follow root_dir/<object>/<image>
       structure. Directory name would be object name.

    Args:
        imgs(list): List of images being used for synthesis 
    Returns:
        list: List of labels/object names corresponding to each image
    '''
    labels = []
    for img_file in imgs:
        label = img_file.split('/')[-2]
        labels.append(label)
    return labels

def get_annotation_from_mask_file(mask_file, scale=1.0):
    '''Given a mask file and scale, return the bounding box annotations

    Args:
        mask_file(string): Path of the mask file
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    if os.path.exists(mask_file):
        mask = cv2.imread(mask_file)
        if INVERTED_MASK:
            mask = 255 - mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if len(np.where(rows)[0]) > 0:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            return int(scale*xmin), int(scale*xmax), int(scale*ymin), int(scale*ymax)
        else:
            return -1, -1, -1, -1
    else:
        print ("%s not found. Using empty mask instead." % mask_file)
        return -1, -1, -1, -1

def get_annotation_from_mask(mask):
    '''Given a mask, this returns the bounding box annotations

    Args:
        mask(NumPy Array): Array with the mask
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if len(np.where(rows)[0]) > 0:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return xmin, xmax, ymin, ymax
    else:
        return -1, -1, -1, -1

def write_imageset_file(exp_dir, img_files, anno_files):
    '''Writes the imageset file which has the generated images and corresponding annotation files
       for a given experiment

    Args:
        exp_dir(string): Experiment directory where all the generated images, annotation and imageset
                         files will be stored
        img_files(list): List of image files that were generated
        anno_files(list): List of annotation files corresponding to each image file
    '''
    with open(os.path.join(exp_dir,'train.txt'),'w') as f:
        for i in range(len(img_files)):
            f.write('%s %s\n'%(img_files[i], anno_files[i]))

def write_labels_file(exp_dir, labels):
    '''Writes the labels file which has the name of an object on each line

    Args:
        exp_dir(string): Experiment directory where all the generated images, annotation and imageset
                         files will be stored
        labels(list): List of labels. This will be useful while training an object detector
    '''
    unique_labels = ['__background__'] + sorted(set(labels))
    with open(os.path.join(exp_dir,'labels.txt'),'w') as f:
        for i, label in enumerate(unique_labels):
            f.write('%s %s\n'%(i, label))

def keep_selected_labels(img_files, labels):
    '''Filters image files and labels to only retain those that are selected. Useful when one doesn't 
       want all objects to be used for synthesis

    Args:
        img_files(list): List of images in the root directory
        labels(list): List of labels corresponding to each image
    Returns:
        new_image_files(list): Selected list of images
        new_labels(list): Selected list of labels corresponidng to each imahe in above list
    '''
    with open(SELECTED_LIST_FILE) as f:
        selected_labels = [x.strip() for x in f.readlines()]
    new_img_files = []
    new_labels = []
    for i in range(len(img_files)):
        if labels[i] in selected_labels:
            new_img_files.append(img_files[i])
            new_labels.append(labels[i])
    return new_img_files, new_labels

def PIL2array1C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0])

def PIL2array3C(img):
    '''Converts a PIL image to NumPy Array

    Args:
        img(PIL Image): Input PIL image
    Returns:
        NumPy Array: Converted image
    '''
    return np.array(img.getdata(), np.uint8).reshape(img.size[1], img.size[0], 3)

#['none']
def create_image_anno(blend_json, blending_list=BLENDING_LIST):
    """
    if 'none' not in img_file:
        return 
    
    print ("Working on %s" % img_file)
    if os.path.exists(anno_file):
        return anno_file
    """
    #all_objects = objects + distractor_objects
    bg = blend_json['bg']
    #['inst_path']
    #['semantic_path']

    #print("bgf")
    #print(bg_file)
    try:
        im_path = bg['im_path']
    except KeyError:
        print("no bg im_path")
        return

    try:
        inst_path = bg['inst_path']
        bg_inst = Image.open(inst_path)
    except KeyError:
        print("no bg inst_path")
        bg_inst = None

    try:
        segm_path = bg['segm_path']
        bg_segm = Image.open(segm_path)
    except KeyError:
        print("no bg segm_path")
        bg_segm = None

    all_objects = blend_json['fgs']

    top = Element('annotation')
    bg_img = Image.open(im_path)
    bgs = []
    for i in range(len(blending_list)):
        bgs.append(bg_img.copy())

    for idx, obj in enumerate(all_objects):
        x, y = obj['x'], obj['y']

        print("fgf")
        print(obj['path'])
        fg = Image.open(obj['path'])
        xmin, xmax, ymin, ymax = get_annotation_from_mask_file(get_mask_file(obj['path']))

        fg = fg.crop((xmin, ymin, xmax, ymax))
        orig_w, orig_h = fg.size
        mask_file =  get_mask_file(obj['path'])
        print("mask")
        print(mask_file)
        mask = Image.open(mask_file)

        mask = mask.crop((xmin, ymin, xmax, ymax))
        if INVERTED_MASK:
            mask = Image.fromarray(255 - PIL2array1C(mask))
        o_w, o_h = orig_w, orig_h

        #mirror
        if obj['mirror'] == True:
            fg = ImageOps.mirror(fg)
            mask = ImageOps.mirror(mask)

        #scale
        # TODO check if > 1?
        scale = obj['scale']
        o_w, o_h = int(scale*orig_w), int(scale*orig_h)
        
        fg = fg.resize((o_w, o_h), Image.ANTIALIAS)
        mask = mask.resize((o_w, o_h), Image.ANTIALIAS)

        #rotate       
        angle = obj['angle']
        print(angle)
        fg_tmp = fg.rotate(angle) #, expand=True)
        mask_tmp = mask.rotate(angle) #, expand=True)
        o_w, o_h = fg_tmp.size
        mask = mask_tmp
        fg = fg_tmp

        xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)

        x = x - o_w // 2
        y = y - o_h // 2
        for i in range(len(blending_list)):
            if blending_list[i] == 'none' or blending_list[i] == 'motion':
                bgs[i].paste(fg, (x, y), mask)
            elif blending_list[i] == 'poisson':
                offset = (y, x)
                img_mask = PIL2array1C(mask)
                img_src = PIL2array3C(fg).astype(np.float64)
                img_target = PIL2array3C(bgs[i])
                img_mask, img_src, offset_adj \
                       = create_mask(img_mask.astype(np.float64), img_target, img_src, offset=offset)
                bg_array = poisson_blend(img_mask, img_src, img_target, method='normal', offset_adj=offset_adj)
                bgs[i] = Image.fromarray(bg_array, 'RGB') 
            elif blending_list[i] == 'gaussian':
                bgs[i].paste(fg, (x, y), Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask),(5,5),2)))
            elif blending_list[i] == 'box':
                bgs[i].paste(fg, (x, y), Image.fromarray(cv2.blur(PIL2array1C(mask),(3,3))))
        
        if bg_inst is not None:
            objID = obj['obj_id'] + idx*10
            bg_inst.paste(objID, (x, y), mask)

        if bg_segm is not None:
            classID = obj['class_id']
            bg_segm.paste(classID, (x, y), mask)

        # obj box anno
        object_root = SubElement(top, 'object')
        object_type = obj['class_name']
        object_type_entry = SubElement(object_root, 'name')
        object_type_entry.text = str(object_type)
        object_bndbox_entry = SubElement(object_root, 'bndbox')
        x_min_entry = SubElement(object_bndbox_entry, 'xmin')
        x_min_entry.text = '%d'%(max(1,x+xmin))
        x_max_entry = SubElement(object_bndbox_entry, 'xmax')
        x_max_entry.text = '%d'%(min(o_w,x+xmax))
        y_min_entry = SubElement(object_bndbox_entry, 'ymin')
        y_min_entry.text = '%d'%(max(1,y+ymin))
        y_max_entry = SubElement(object_bndbox_entry, 'ymax')
        y_max_entry.text = '%d'%(min(o_h,y+ymax))
        difficult_entry = SubElement(object_root, 'difficult')
        difficult_entry.text = '0' # Add heuristic to estimate difficulty later on

    for i in range(len(blending_list)):
        if blending_list[i] == 'motion':
            bgs[i] = LinearMotionBlur3C(PIL2array3C(bgs[i]))
        bgs[i].save('./0_' + blending_list[i] + '.png')

    if bg_inst is not None:
        bg_inst.save('./0_inst.png')
    
    if bg_segm is not None:
        bg_inst.save('./0_segm.png')

    """
    # obj box anno
    xmlstr = xml.dom.minidom.parseString(tostring(top)).toprettyxml(indent="    ")
    with open(anno_file, "w") as f:
        f.write(xmlstr)
    """

def gen_syn_data(blend_json):
 
    partial_func = partial( create_image_anno, 
                            blending_list=BLENDING_LIST) 

    p = Pool(NUMBER_OF_WORKERS, init_worker)
    try:
        p.map(partial_func, blend_json)
    except KeyboardInterrupt:
        print ("....\nCaught KeyboardInterrupt, terminating workers")
        p.terminate()
    else:
        p.close()
    p.join()
    
    write_imageset_file('./EXP', list(img_files), list(anno_files))
    #return list(img_files), list(anno_files)

def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
