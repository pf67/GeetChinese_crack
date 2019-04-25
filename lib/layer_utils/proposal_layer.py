# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from lib.utils.bbox_transform import bbox_transform_inv, clip_boxes
from lib.config import config as cfg
from lib.utils.nms_wrapper import nms


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
    """A simplified version compared to fast/er RCNN
       For details please see the technical report
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')

    if cfg_key == "TRAIN":
        pre_nms_topN = cfg.FLAGS.rpn_train_pre_nms_top_n#('rpn_train_pre_nms_top_n', 12000,
        post_nms_topN = cfg.FLAGS.rpn_train_post_nms_top_n#rpn_train_post_nms_top_n', 2000
        nms_thresh = cfg.FLAGS.rpn_train_nms_thresh#'rpn_train_nms_thresh', 0.7
    else:
        pre_nms_topN = cfg.FLAGS.rpn_test_pre_nms_top_n
        post_nms_topN = cfg.FLAGS.rpn_test_post_nms_top_n
        nms_thresh = cfg.FLAGS.rpn_test_nms_thresh

    im_info = im_info[0]
    # Get the scores and bounding boxes
    scores = rpn_cls_prob[:, :, :, num_anchors:]#[,H,W，2*num_anchors]---#18个channel 按照（bg，fg）这里只取了fg的9个channel（（1,9，h,w））
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    rpn_bbox_pred = rpn_bbox_pred.reshape((-1, 4))#(, height, width, A * 4)
    scores = scores.reshape((-1, 1))
    ## #计算经过偏移后的预测坐标
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    # 2. clip predicted boxes to image  将预测框剪切到图像范围内
    proposals = clip_boxes(proposals, im_info[:2])

    # Pick the top region proposals
    # ravel()平铺扁平化  argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出 [::-1]反向取索引
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
        # 取出值scores最大的 pre_nms_topN个
    proposals = proposals[order, :]
    scores = scores[order]

    # Non-maximal suppression
    keep = nms(np.hstack((proposals, scores)), nms_thresh)

    # Pick th top region proposals after NMS
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Only support single image as input
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

    return blob, scores
