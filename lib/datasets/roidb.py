# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import PIL

def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """
  roidb = imdb.roidb
  if not (imdb.name.startswith('coco')):#startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，否则返回 False
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size#获取图片尺寸
         for i in range(imdb.num_images)]
  for i in range(len(imdb.image_index)):
    roidb[i]['image'] = imdb.image_path_at(i)
    if not (imdb.name.startswith('coco')):
      roidb[i]['width'] = sizes[i][0]
      roidb[i]['height'] = sizes[i][1]
    # need gt_overlaps as a dense array for argmax
    #csr_matrix.toarray(order=None, out=None)[source] ：Return a dense ndarray representation of this matrix.
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()#overlaps[ix, cls] = 1.0 ix 图像中物体的序列 cls 图像分类
    print(len(imdb.image_index))
    print(roidb[i]['gt_overlaps'])
    print(gt_overlaps)
    print(roidb[i]['boxes'])
    print(roidb[i]['gt_classes'])
    # max overlap with gt over classes (columns)  某个物体的最大图像cls
    max_overlaps = gt_overlaps.max(axis=1)#[ix,1]
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)#[ix,1]
    roidb[i]['max_classes'] = max_classes
    roidb[i]['max_overlaps'] = max_overlaps
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)#对于读取GT文件，所有都是fg
    nonzero_inds = np.where(max_overlaps > 0)[0]
    print(imdb.image_index[i])
    assert all(max_classes[nonzero_inds] != 0)
