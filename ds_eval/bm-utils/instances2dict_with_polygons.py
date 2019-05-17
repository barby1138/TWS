#!/usr/bin/python
#
# Convert instances from png files to a dictionary
# This files is created according to https://github.com/facebookresearch/Detectron/issues/111

from __future__ import print_function, absolute_import, division
import os, sys

sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )

# Cityscapes imports
from cityscapesscripts.evaluation.instance import *
from cityscapesscripts.helpers.csHelpers import *
import cv2
from maskrcnn_benchmark.utils import cv2_util

def kitti_to_cityscapes_instaces(instance_img):
        kitti_semantic = instance_img // 256
        kitti_instance = instance_img %  256
        #print(kitti_semantic.max())
        #print(kitti_instance.max())

        instance_mask = (kitti_instance > 0)
        cs_instance = (kitti_semantic*1000 + kitti_instance)*instance_mask + kitti_semantic*(1-instance_mask) 
        return cs_instance

def vkitti_to_cityscapes_instaces(instance_img, color_to_instance_map):
        #print(instance_img.shape)
        cs_instance = np.zeros((instance_img.shape[0], instance_img.shape[1]), dtype=np.uint32)
        
        for x in range(0,instance_img.shape[0]):
            for y in range(0,instance_img.shape[1]):
                #print(instance_img[x][y])
                try:
                    #print(tuple(instance_img[x][y]))
                    #print(color_to_instance_map[ '#%02x%02x%02x' % tuple(instance_img[x][y]) ])
                    cs_instance[x][y] = color_to_instance_map[ '#%02x%02x%02x' % tuple(instance_img[x][y]) ]
                except KeyError:
                    #print('warning unknown color')
                    pass
        
        #print(instance_img)
        #cs_instance = color_to_instance_map[ '#%02x%02x%02x' % tuple(map(tuple, instance_img)) ] #tuple(instance_img) ]
        #print(cs_instance)
        return cs_instance

def instances2dict_with_polygons(imageFileList, instance_format='cityscapes', color_to_instance_map=None, verbose=False):
    imgCount     = 0
    instanceDict = {}

    if not isinstance(imageFileList, list):
        imageFileList = [imageFileList]

    if verbose:
        print("Processing {} images...".format(len(imageFileList)))

    for imageFileName in imageFileList:
        # Load image
        img = Image.open(imageFileName)

        # Image as numpy array
        imgNp = np.array(img)
        if instance_format == 'kitti' or instance_format == 'tws':
            imgNp = kitti_to_cityscapes_instaces(imgNp)
        if instance_format == 'vkitti':
            imgNp = vkitti_to_cityscapes_instaces(imgNp, color_to_instance_map)

        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp):
            #print(instanceId)
            if instanceId < 1000:
                continue
            instanceObj = Instance(imgNp, instanceId)
            instanceObj_dict = instanceObj.toDict()

            #instances[id2label[instanceObj.labelID].name].append(instanceObj.toDict())
            if id2label[instanceObj.labelID].hasInstances:
                #print(imgNp.dtype)
                #print(instanceId.dtype)
                mask = (imgNp == instanceId).astype(np.uint8)
                #print(mask.dtype)
                #print(mask)
                #print(mask.shape)
                #print(imgNp.shape)
                contour, hier = cv2_util.findContours(
                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                polygons = [c.reshape(-1).tolist() for c in contour]
                instanceObj_dict['contours'] = polygons

            instances[id2label[instanceObj.labelID].name].append(instanceObj_dict)

        imgKey = os.path.abspath(imageFileName)
        instanceDict[imgKey] = instances
        imgCount += 1

        if verbose:
            print("\rImages Processed: {}".format(imgCount), end=' ')
            sys.stdout.flush()

    if verbose:
        print("")

    return instanceDict

def main(argv):
    fileList = []
    if (len(argv) > 2):
        for arg in argv:
            if ("png" in arg):
                fileList.append(arg)
    instances2dict_with_polygons(fileList, True)

if __name__ == "__main__":
    main(sys.argv[1:])
