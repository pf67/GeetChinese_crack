#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms
from lib.utils.test import im_detect
#from nets.resnet_v1 import resnetv1
from lib.nets.vgg16 import vgg16
from lib.utils.timer import Timer

CLASSES = ('__background__',
           'chinese')
#40000
NETS = {'vgg16': ('vgg16_faster_rcnn_iter_25000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('GeetCode_2019_train',)}


def getimgth(image):
    height = image.shape[0]
    width = image.shape[1]
    # print(image.shape)
    img = image[int(height / 4):int(height * 3 / 4), int(width / 4):int(width * 3 / 4)]
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if  ret < 127:
        if ret - 20 > 0:
            ret = ret - 20
    else:
        if ret + 20 < 255:
            ret = ret + 20

    return ret

def getbboximage(bbox,im,image_name,index,output_path):

     x1 = int(bbox[0]+0.5)
     y1 = int(bbox[1]+0.5)
     x2 = int(bbox[2]+0.5)
     y2 = int(bbox[3]+0.5)

     region = im[y1:y2,x1:x2]
     region =  cv2.cvtColor(region,cv2.COLOR_BGR2GRAY)
     ret =  getimgth(region)
     ret, region = cv2.threshold(region, ret, 255, cv2.THRESH_BINARY)

     out_file = output_path + image_name.split('.')[0] + '_' + str(index) + '.jpg'
     cv2.imwrite(out_file, region)
     return out_file,bbox



def vis_detections(im, class_name, dets,image_name,output_path, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    #    # python-opencv 中读取图片默认保存为[w,h,channel](w,h顺序不确定)
    # 其中 channel：BGR 存储，而画图时，需要按RGB格式，因此此处作转换。
    im = im[:, :, (2, 1, 0)]
    chinese_bbox= []

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        chinese_bbox.append(getbboximage(bbox,im,image_name,i,output_path))
    return chinese_bbox


def boxdetect(sess, net,im_file,output_path):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the image
    im_file = im_file.replace('\\', '/')
    im = cv2.imread(im_file)
    image_name = im_file.split(r'/')[-1]
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.1
    NMS_THRESH = 0.1
    geetcode_bbox=[]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        bbox = vis_detections(im, cls, dets,image_name,output_path, thresh=CONF_THRESH)
        geetcode_bbox.append(bbox)
    return geetcode_bbox

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN')
    parser.add_argument('--net', dest='detect_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [GeetCode_2019_train]',
                        choices=DATASETS.keys(), default='pascal_voc')
    args = parser.parse_args()

    return args

def getimgnamepath(img_path):
    img_name=[]
    for root, sub_folder, name_list in os.walk(img_path):
        img_name += [os.path.join(root, name) for name in name_list]
    return img_name

def load_faster_rcnn_network():
    args = parse_args()
    detect_net = args.detect_net
    dataset = args.dataset
    tfmodel = os.path.join('output', detect_net, DATASETS[dataset][0], 'default', NETS[detect_net][0])
    print(tfmodel)
    if not os.path.isfile(tfmodel + '.meta'):
        print(tfmodel)
        raise IOError(('{:s} not found.\n').format(tfmodel + '.meta'))

    fasterrcnn_graph = tf.Graph()
    with fasterrcnn_graph.as_default():
        # set config
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        # init session
        sess = tf.Session(graph=fasterrcnn_graph,config=tfconfig)
        # load network
        if detect_net == 'vgg16':
            net = vgg16(batch_size=1)
        # elif demonet == 'res101':
        # net = resnetv1(batch_size=1, num_layers=101)
        else:
            raise NotImplementedError
        net.create_architecture(sess, "TEST", len(CLASSES),
                                tag='default', anchor_scales=[8, 16, 32])
        saver = tf.train.Saver()
        saver.restore(sess, tfmodel)
        print('Loaded network {:s}'.format(tfmodel))

    return sess,net


def run_geetcode_boxdetect(im,output_path, sess, net):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('detect_net for data/demo/{}'.format(im))
    geetcode_bbox = boxdetect(sess, net,im,output_path)

    # plt.show()
    return geetcode_bbox


if __name__ == '__main__':
    args = parse_args()