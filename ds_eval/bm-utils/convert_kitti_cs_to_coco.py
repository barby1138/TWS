
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import h5py
import json
import os
import scipy.misc
import sys

import instances2dict_with_polygons as cs

import numpy as np

from shutil import copyfile

def parse_args():
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--dataset', help="kitti, vkitti, cityscapes", default=None, type=str)
    parser.add_argument(
        '--outdir', help="output dir for ds files", default=None, type=str)
    parser.add_argument(
        '--datadir', help="data dir for annotations to be converted",
        default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def poly_to_box(poly):
    """Convert a polygon into a tight bounding box."""
    x0 = min(min(p[::2]) for p in poly)
    x1 = max(max(p[::2]) for p in poly)
    y0 = min(min(p[1::2]) for p in poly)
    y1 = max(max(p[1::2]) for p in poly)
    box_from_poly = [x0, y0, x1, y1]

    return box_from_poly

def xyxy_to_xywh(xyxy_box):
    xmin, ymin, xmax, ymax = xyxy_box
    TO_REMOVE = 1
    xywh_box = (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE)
    return xywh_box

# for Cityscapes
def getLabelID(self, instID):
    if (instID < 1000):
        return instID
    else:
        return int(instID / 1000)

def make_dirs(ds_name, out_dir):
    out_dir_json = os.path.join(out_dir, ds_name, 'annotations')
    out_dir_images = os.path.join(out_dir, ds_name, 'images')
    os.mkdir(os.path.join(out_dir,ds_name))
    os.mkdir(out_dir_json)
    os.mkdir(out_dir_images)
    return out_dir_json, out_dir_images

def make_ds_file_names(root, ds_name, data_set, filename, prefix=None):
    if ds_name == 'cityscapes':
        ends_in = '%s_polygons.json'
        ds_file_name = filename[:-len(ends_in % data_set.split('_')[0])] + 'leftImg8bit.png'
        ds_seg_file_name = filename[:-len(ends_in % data_set.split('_')[0])] + '%s_instanceIds.png' % data_set.split('_')[0]
        # TODO
        full_seg_file_name = os.path.join(root, ds_seg_file_name)
        full_rgb_file_name = os.path.join(root, ds_file_name).replace('gtFine_trainvaltest/gtFine', 'leftImg8bit_trainvaltest/leftImg8bit')
    elif ds_name == 'kitti':
        ds_file_name, ds_seg_file_name = filename, filename
        full_rgb_file_name = os.path.join(root, filename)
        full_seg_file_name = full_rgb_file_name.replace('/image_2/', '/instance/')
    elif ds_name == 'vkitti':
        ds_file_name = filename
        if prefix is not None:
            ds_file_name = prefix + ds_file_name
        else:
            # todo ex?
            print('Error!!! prefix needed')
        ds_seg_file_name = filename

        full_rgb_file_name = os.path.join(root, filename)
        full_seg_file_name = full_rgb_file_name.replace('/vkitti_1.3.1_rgb/', '/vkitti_1.3.1_scenegt/')
    else:
        # todo ex?
        ds_file_name, ds_seg_file_name, full_rgb_file_name, full_seg_file_name = None, None, None, None

    return ds_file_name, ds_seg_file_name, full_rgb_file_name, full_seg_file_name

def process_file(   root,
                    out_dir_images,
                    ds_name, 
                    ds_subname, 
                    # for same named files under subfolders in initial DS (vkitti)
                    prefix, 
                    #if seg is rgb (vkitti)
                    color_to_instance_map, 
                    filename, 
                    images,
                    annotations, 
                    category_dict,
                    category_instancesonly, 
                    img_id, 
                    ann_id, 
                    cat_id):

            image = {}
            image['id'] = img_id
            img_id += 1

            # TODO review
            w, h = 1242, 375 # kitti, vkitti
            if ds_name == 'cityscapes':
                w, h = 2048, 1024
            image['width'] = w
            image['height'] = h

            image['file_name'], image['seg_file_name'], fullname_rgb, fullname = make_ds_file_names(root, ds_name, ds_subname, filename, prefix=prefix)
            images.append(image)

            dst = os.path.join(out_dir_images, image['file_name'])
            copyfile(fullname_rgb, dst)
            
            objects = cs.instances2dict_with_polygons([fullname], 
                                                    instance_format=ds_name, 
                                                    color_to_instance_map=color_to_instance_map, 
                                                    verbose=False)[fullname]
            #print(objects)
            for object_cls in objects:
                if object_cls not in category_instancesonly:
                            #print('Warning: object_cls not in cat. %s' % object_cls)
                            continue  # skip non-instance categories

                for obj in objects[object_cls]:
                    if obj['contours'] == []:
                        print('Warning: empty contours. %s' % object_cls)
                        continue  # skip non-instance categories

                    len_p = [len(p) for p in obj['contours']]
                    if min(len_p) <= 4:
                        #print('Warning: invalid contours. %s' % object_cls)
                        continue  # skip non-instance categories

                    ann = {}
                    ann['id'] = ann_id
                    ann_id += 1
                    ann['image_id'] = image['id']
                    ann['segmentation'] = obj['contours']

                    if object_cls not in category_dict:
                        category_dict[object_cls] = cat_id
                        cat_id += 1
                    ann['category_id'] = category_dict[object_cls]
                    ann['iscrowd'] = 0
                    ann['area'] = obj['pixelCount']
                            
                    xyxy_box = poly_to_box(ann['segmentation'])
                    xywh_box = xyxy_to_xywh(xyxy_box)
                    ann['bbox'] = xywh_box

                    annotations.append(ann)
            
            return img_id, ann_id, cat_id

def convert_cityscapes_instance_only(
        data_dir, out_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    sets = [
        'gtFine_val',
        'gtFine_train',
        #'gtFine_test',

        # 'gtCoarse_train',
        # 'gtCoarse_val',
        # 'gtCoarse_train_extra'
    ]
    ann_dirs = [
        'gtFine_trainvaltest/gtFine/val',
        'gtFine_trainvaltest/gtFine/train',
        #'gtFine_trainvaltest/gtFine/test',

        # 'gtCoarse/train',
        # 'gtCoarse/train_extra',
        # 'gtCoarse/val'
    ]
    json_name = 'instancesonly_filtered_%s.json'
    ends_in = '%s_polygons.json'
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {}

    category_instancesonly = [
        #'person',
        #'rider',
        'car',
        
        #'truck',
        #'bus',
        #'train',
        #'motorcycle',
        #'bicycle',
    ]

    out_dir_json, out_dir_images = make_dirs('cityscapes', out_dir)

    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        ann_dir = os.path.join(data_dir, ann_dir)

        for root, _, files in os.walk(ann_dir):

            for filename in files:
                if filename.endswith(ends_in % data_set.split('_')[0]):
                    if len(images) % 50 == 0:
                        print("Processed %s images, %s annotations" % (len(images), len(annotations)))

                    img_id, ann_id, cat_id = process_file(root=root,
                            out_dir_images=out_dir_images,
                            ds_name='cityscapes', 
                            ds_subname=data_set, 
                            prefix=None, 
                            color_to_instance_map=None, 
                            filename=filename, 
                            images=images,
                            annotations=annotations, 
                            category_dict=category_dict,
                            category_instancesonly=category_instancesonly, 
                            img_id=img_id, 
                            ann_id=ann_id, 
                            cat_id=cat_id)

        ann_dict['images'] = images
        categories = [{"id": category_dict[name], "name": name} for name in
                      category_dict]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(out_dir_json, json_name % data_set), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))

# KITTI segm is in CS format (almost - instance id format differs) but different paths
def convert_kitti_instance_only(data_dir, out_dir):

    sets = [
        'kitti_training',
    ]
    ann_dirs = [
        'data_semantics/training/image_2',
    ]
    json_name = 'instancesonly_filtered_%s.json'
    ends_in = '%s_polygons.json'
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {}

    category_instancesonly = [
        #'person',
        #'rider',
        'car',

        #'truck',
        #'bus',
        #'train',
        #'motorcycle',
        #'bicycle',
    ]

    out_dir_json, out_dir_images = make_dirs('kitti', out_dir)

    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        ann_dir = os.path.join(data_dir, ann_dir)

        for root, _, files in os.walk(ann_dir):
            
            for filename in files:
                if len(images) % 50 == 0:
                    print("Processed %s images, %s annotations" % (len(images), len(annotations)))

                img_id, ann_id, cat_id = process_file(root=root,
                            out_dir_images=out_dir_images,
                            ds_name='kitti', 
                            ds_subname=data_set, 
                            prefix=None, 
                            color_to_instance_map=None, 
                            filename=filename, 
                            images=images,
                            annotations=annotations, 
                            category_dict=category_dict,
                            category_instancesonly=category_instancesonly, 
                            img_id=img_id, 
                            ann_id=ann_id, 
                            cat_id=cat_id)

        ann_dict['images'] = images
        categories = [{"id": category_dict[name], "name": name} for name in
                      category_dict]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(out_dir_json, json_name % data_set), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))

def convert_vkitti_instance_only(
        data_dir, out_dir):

    def create_vkitti_color_to_instance_map(filename):
        #vkitti_color_to_instance_map[]
        colormap = {}

        import csv
        with open(filename, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for idx, row in enumerate(spamreader):
                #print(', '.join(row))
                #print(idx)
                if idx == 0:
                    continue

                category_id, r, g, b = row[0], row[1], row[2], row[3]
                arr = category_id.split(':')
                inst_id = 0
                cat = arr[0]
                cat_id = 0
                if len(arr) > 1:
                    inst_id = int(arr[1])
                
                if cat == 'Car':
                    cat_id = 26

                # 26*1000 + 0                  
                colormap[ '#%02x%02x%02x' % (int(r), int(g), int(b)) ] = cat_id * 1000 + inst_id
                #print(colormap[ '#%02x%02x%02x' % (int(r), int(g), int(b)) ])
        
        return colormap

    sets = [
        'vkitti_training',
    ]
    ann_dirs = [
        'vkitti_1.3.1_rgb',
    ]
    worlds = [
        '0001',
        '0002',
        '0006',
        '0018',
        '0020',
    ]
    variations = [
        '15-deg-left',
        '15-deg-right',
        '30-deg-left',
        '30-deg-right',

        'clone',
        
        'fog',
        'morning',
        'overcast',
        'rain',
        'sunset',
    ]

    json_name = 'instancesonly_filtered_%s.json'
    ends_in = '%s_polygons.json'
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {}

    category_instancesonly = [
        #'Building', 
        'car', #'Car'
        #'GuardRail', 
        #'Misc', 
        #'Pole', 
        #'Road', 
        #'Sky', 
        #'Terrain', 
        #'TrafficLight', 
        #'TrafficSign', 
        #'Tree', 
        #'Truck', 
        #'Van', 
        #'Vegetation'
    ]

    out_dir_json, out_dir_images = make_dirs('vkitti', out_dir)

    for data_set, ann_dir_0 in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        ann_dict = {}
        images = []
        annotations = []
        #ann_dir = os.path.join(data_dir, ann_dir)

        for world in worlds:
            for variation in variations:
                ann_dir = os.path.join(data_dir, ann_dir_0, world, variation)
                for root, _, files in os.walk(ann_dir):
            
                    seg_root = os.path.join(data_dir, 'vkitti_1.3.1_scenegt')
                    # 0001_clone_scenegt_rgb_encoding.txt
                    colour_file = world + '_' + variation + '_scenegt_rgb_encoding.txt'
                    colour_file_path = os.path.join(seg_root, colour_file)
                    vkitti_color_to_instance_map = create_vkitti_color_to_instance_map(colour_file_path)

                    for filename in files:
                        if len(images) % 50 == 0:
                            print("Processed %s images, %s annotations" % (len(images), len(annotations)))

                        img_id, ann_id, cat_id = process_file(root=root,
                            out_dir_images=out_dir_images,
                            ds_name='vkitti', 
                            ds_subname=data_set, 
                            prefix=world + '_' + variation + '_', 
                            color_to_instance_map=vkitti_color_to_instance_map, 
                            filename=filename, 
                            images=images,
                            annotations=annotations, 
                            category_dict=category_dict,
                            category_instancesonly=category_instancesonly, 
                            img_id=img_id, 
                            ann_id=ann_id, 
                            cat_id=cat_id)

        ann_dict['images'] = images
        categories = [{"id": category_dict[name], "name": name} for name in category_dict]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))
        with open(os.path.join(out_dir_json, json_name % data_set), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))


if __name__ == '__main__':
    args = parse_args()
    if args.dataset == "cityscapes_instance_only":
        convert_cityscapes_instance_only(args.datadir, args.outdir)
    elif args.dataset == "kitti_instance_only":
        convert_kitti_instance_only(args.datadir, args.outdir)
    elif args.dataset == "vkitti_instance_only":
        convert_vkitti_instance_only(args.datadir, args.outdir)
    else:
        print("Dataset not supported: %s" % args.dataset)

