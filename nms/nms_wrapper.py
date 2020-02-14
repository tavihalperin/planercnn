# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import torchvision

def nms(dets, thresh):
  """Dispatch to either CPU or GPU NMS implementations.
  Accept dets as tensor"""
  return torchvision.ops.nms(dets[:, :4], dets[:, 4], thresh)

