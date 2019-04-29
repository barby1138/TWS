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

from collections import namedtuple

import pb as pb

import math
from pyblur.pyblur import *
from collections import namedtuple

import random as rnd

######################
#import numpy as np
#import cv2

def color_transfer(source, target, clip=True, preserve_paper=True):
	"""
	Transfers the color distribution from the source to the target
	image using the mean and standard deviations of the L*a*b*
	color space.
	This implementation is (loosely) based on to the "Color Transfer
	between Images" paper by Reinhard et al., 2001.
	Parameters:
	-------
	source: NumPy array
		OpenCV image in BGR color space (the source image)
	target: NumPy array
		OpenCV image in BGR color space (the target image)
	clip: Should components of L*a*b* image be scaled by np.clip before 
		converting back to BGR color space?
		If False then components will be min-max scaled appropriately.
		Clipping will keep target image brightness truer to the input.
		Scaling will adjust image brightness to avoid washed out portions
		in the resulting color transfer that can be caused by clipping.
	preserve_paper: Should color transfer strictly follow methodology
		layed out in original paper? The method does not always produce
		aesthetically pleasing results.
		If False then L*a*b* components will scaled using the reciprocal of
		the scaling factor proposed in the paper.  This method seems to produce
		more consistently aesthetically pleasing results 
	Returns:
	-------
	transfer: NumPy array
		OpenCV image (w, h, 3) NumPy array (uint8)
	"""
	# convert the images from the RGB to L*ab* color space, being
	# sure to utilizing the floating point data type (note: OpenCV
	# expects floats to be 32-bit, so use that instead of 64-bit)
	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

	# compute color statistics for the source and target images
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

	# subtract the means from the target image
	(l, a, b) = cv2.split(target)
	l -= lMeanTar
	a -= aMeanTar
	b -= bMeanTar

	if preserve_paper:
		# scale by the standard deviations using paper proposed factor
		l = (lStdTar / lStdSrc) * l
		a = (aStdTar / aStdSrc) * a
		b = (bStdTar / bStdSrc) * b
	else:
		# scale by the standard deviations using reciprocal of paper proposed factor
		l = (lStdSrc / lStdTar) * l
		a = (aStdSrc / aStdTar) * a
		b = (bStdSrc / bStdTar) * b

	# add in the source mean
	l += lMeanSrc
	a += aMeanSrc
	b += bMeanSrc

	# clip/scale the pixel intensities to [0, 255] if they fall
	# outside this range
	l = _scale_array(l, clip=clip)
	a = _scale_array(a, clip=clip)
	b = _scale_array(b, clip=clip)

	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data
	# type
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
	
	# return the color transferred image
	return transfer

def image_stats(image):
	"""
	Parameters:
	-------
	image: NumPy array
		OpenCV image in L*a*b* color space
	Returns:
	-------
	Tuple of mean and standard deviations for the L*, a*, and b*
	channels, respectively
	"""
	# compute the mean and standard deviation of each channel
	(l, a, b) = cv2.split(image)
	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())

	# return the color statistics
	return (lMean, lStd, aMean, aStd, bMean, bStd)

def _min_max_scale(arr, new_range=(0, 255)):
	"""
	Perform min-max scaling to a NumPy array
	Parameters:
	-------
	arr: NumPy array to be scaled to [new_min, new_max] range
	new_range: tuple of form (min, max) specifying range of
		transformed array
	Returns:
	-------
	NumPy array that has been scaled to be in
	[new_range[0], new_range[1]] range
	"""
	# get array's current min and max
	mn = arr.min()
	mx = arr.max()

	# check if scaling needs to be done to be in new_range
	if mn < new_range[0] or mx > new_range[1]:
		# perform min-max scaling
		scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
	else:
		# return array if already in range
		scaled = arr

	return scaled

def _scale_array(arr, clip=True):
	"""
	Trim NumPy array values to be in [0, 255] range with option of
	clipping or scaling.
	Parameters:
	-------
	arr: array to be trimmed to [0, 255] range
	clip: should array be scaled by np.clip? if False then input
		array will be min-max scaled to range
		[max([arr.min(), 0]), min([arr.max(), 255])]
	Returns:
	-------
	NumPy array that has been scaled to be in [0, 255] range
	"""
	if clip:
		scaled = np.clip(arr, 0, 255)
	else:
		scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
		scaled = _min_max_scale(arr, new_range=scale_range)

	return scaled
######################

# TODO move to cs
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

#set in cmd json?
MAX_BLEND_FG = 5
MAX_MIN_FG_DIFF = 1 # min = max - diff

#POISSON_BLENDING_DIR = './EXP'
INVERTED_MASK = False
NUMBER_OF_WORKERS = 4

ALPHA = 0.7
BLENDING_VARIATIONS = 2
# TODO investigate poisson and motion
#BLENDING_LIST = ['poisson']
BLENDING_LIST = ['gaussian','none', 'box']
#BLENDING_LIST = ['gaussian','poisson', 'none', 'box', 'motion']

#from defaults import *
#sys.path.insert(0, POISSON_BLENDING_DIR)

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
def create_image_anno(blend_json, blending_variations=BLENDING_VARIATIONS):
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

    blending_list = BLENDING_LIST

    try:
        im_path = bg['im_path']
    except KeyError:
        print("no bg im_path")
        return

    try:
        inst_path = bg['inst_path']
        bg_inst = Image.open(inst_path)
        bg_inst_np = np.array(bg_inst)
        unique_instIDs = np.unique(bg_inst_np)
        current_obj_id = 0
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
    all_objects_len = len(all_objects)
    max_objects_to_blend_len = MAX_BLEND_FG
    if max_objects_to_blend_len > all_objects_len:
        print("WARNING!!! max_objects_to_blend_len > all_objects_len")
        max_objects_to_blend_len = all_objects_len
    min_objects_to_blend_len = max_objects_to_blend_len - MAX_MIN_FG_DIFF
    if min_objects_to_blend_len < 0:
        print("WARNING!!! min_objects_to_blend_len < 0")
        min_objects_to_blend_len = max_objects_to_blend_len

    top = Element('annotation')
    bg_img = Image.open(im_path)
    hl = Image.new(mode='L', size=bg_img.size, color=int(255*ALPHA))
    bgs = []
    hls = []
    bg_segms = []
    bg_insts = []
    blend_fgs = []
    for i in range(blending_variations):
        bgs.append(bg_img.copy())
        hls.append(hl.copy())
        if bg_segm is not None:
            bg_segms.append(bg_segm.copy())
        if bg_inst is not None:
            bg_insts.append(bg_inst.copy())

        print("max %d min %d" % (max_objects_to_blend_len, min_objects_to_blend_len))
        objects_to_blend_len = np.random.randint(low=min_objects_to_blend_len, high=max_objects_to_blend_len)
        blend_fg = rnd.sample(range(len(all_objects)), objects_to_blend_len)
        print("in pic %d objects_to_blend_len %d" % (i, objects_to_blend_len))
        print(blend_fg)
        # TODO check for same existing fg config
        blend_fgs.append(blend_fg)

    for idx, obj in enumerate(all_objects):
        x, y = obj['x'], obj['y']

        #print("fgf")
        #print(obj['path'])
        fg = Image.open(obj['path'])
        xmin, xmax, ymin, ymax = get_annotation_from_mask_file(get_mask_file(obj['path']))

        fg = fg.crop((xmin, ymin, xmax, ymax))
        orig_w, orig_h = fg.size
        mask_file =  get_mask_file(obj['path'])
        #print("mask")
        #print(mask_file)
        mask = Image.open(mask_file)

        mask = mask.crop((xmin, ymin, xmax, ymax))
        if INVERTED_MASK:
            mask = Image.fromarray(255 - PIL2array1C(mask))
        o_w, o_h = orig_w, orig_h

        #color
        """
        fg_np = color_transfer( np.array(bg_img), np.array(fg) )
        fg = Image.fromarray(fg_np)
        """

        #mirror
        if obj['mirror'] == True:
            fg = ImageOps.mirror(fg)
            mask = ImageOps.mirror(mask)

        #scale
        # TODO check if > 1?
        scale = obj['scale']
        o_w, o_h = int(scale*orig_w), int(scale*orig_h)
        
        fg = fg.resize((o_w, o_h), Image.ANTIALIAS)
        # must not ANTIALIAS
        mask = mask.resize((o_w, o_h))
        
        #rotate       
        angle = obj['angle']
        #print(angle)
        fg_tmp = fg.rotate(angle) #, expand=True)
        mask_tmp = mask.rotate(angle) #, expand=True)
        o_w, o_h = fg_tmp.size
        mask = mask_tmp
        fg = fg_tmp
        
        xmin, xmax, ymin, ymax = get_annotation_from_mask(mask)

        x = x - o_w // 2
        y = y - o_h // 2

        for i in range(blending_variations):
            print(blend_fgs[0])
            if idx not in blend_fgs[i]:
                print("skip obj %d" % idx)
                continue                

            hls[i].paste(255, (x, y), mask)

            if bg_inst is not None:
                classID = name2label[obj['class_name']].id
                while True:
                    objID = classID * 256 + current_obj_id
                    current_obj_id += 1
                    #if objID not in unique_instIDs:
                    if np.all(unique_instIDs != objID):
                        #print(unique_instIDs)
                        #print(objID)
                        unique_instIDs = np.append(unique_instIDs, objID)
                        break

                bg_insts[i].paste(objID, (x, y), mask)

            if bg_segm is not None:
                classID = name2label[obj['class_name']].id
                bg_segms[i].paste(classID, (x, y), mask)

            k = np.random.randint(len(blending_list))
            if blending_list[k] == 'none' or blending_list[k] == 'motion':
                bgs[i].paste(fg, (x, y), mask)
            elif blending_list[k] == 'poisson':
                offset = (y, x)
                img_mask = PIL2array1C(mask)
                img_src = PIL2array3C(fg).astype(np.float64)
                img_target = PIL2array3C(bgs[i])
                img_mask, img_src, offset_adj = pb.create_mask(img_mask.astype(np.float64), img_target, img_src, offset=offset)
                bg_array = pb.blend(img_mask, img_src, img_target, method='normal', offset_adj=offset_adj)
                #bg_array = pb.blend(img_mask, img_src, img_target, offset=offset)
                bgs[i] = Image.fromarray(bg_array, 'RGB') 
            elif blending_list[k] == 'gaussian':
                bgs[i].paste(fg, (x, y), Image.fromarray(cv2.GaussianBlur(PIL2array1C(mask),(5,5),2)))
            elif blending_list[k] == 'box':
                bgs[i].paste(fg, (x, y), Image.fromarray(cv2.blur(PIL2array1C(mask),(3,3))))

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

    for i in range(blending_variations):
        if blending_list[i] == 'motion':
            bgs[i] = LinearMotionBlur3C(PIL2array3C(bgs[i]))
        bgs[i].save('./0_' + str(i) + '.png')

        bgs[i].putalpha(hls[i])
        bgs[i].save('./0_' + str(i) + '_hl.png')

        if bg_inst is not None:
            bg_insts[i].save('./0_' + str(i) + '_inst.png')
    
        if bg_segm is not None:
            bg_segms[i].save('./0_' + str(i) + '_segm.png')
            bg_segm_np = np.array(bg_segms[i])

            bg_segm_rgb_np = np.zeros((bg_segm_np.shape[0], bg_segm_np.shape[1], 3), dtype=np.uint8)
            for x in range(0,bg_segm_np.shape[0]):
                for y in range(0,bg_segm_np.shape[1]):
                    #print(instance_img[x][y])
                    bg_segm_rgb_np[x][y] = id2label[bg_segm_np[x][y]].color

            #bg_segm_rgb_np = id2label[bg_segm_np].color

            bg_segm_rgb = Image.fromarray(bg_segm_rgb_np) #, 'RGB')
            bg_segm_rgb.save('./0_' + str(i) + '_segm_rgb.png')

    """
    # obj box anno
    xmlstr = xml.dom.minidom.parseString(tostring(top)).toprettyxml(indent="    ")
    with open(anno_file, "w") as f:
        f.write(xmlstr)
    """

def gen_syn_data(blend_json):
 
    partial_func = partial( create_image_anno, 
                            blending_variations=BLENDING_VARIATIONS) 

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
