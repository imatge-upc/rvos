# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# ----------------------------------------------------------------------------

import skimage
import numpy as np

def overlay(image,mask,colors=[255,0,0],cscale=2,alpha=0.4):
  """ Overlay segmentation on top of RGB image. """

  colors = np.atleast_2d(colors) * cscale

  im_overlay    = image.copy()
  object_ids = np.unique(mask)

  for object_id in object_ids[1:]:
    # Overlay color on  binary mask

    foreground  = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
    binary_mask = mask == object_id

    # Compose image
    im_overlay[binary_mask] = foreground[binary_mask]

    countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
    im_overlay[countours,:] = 0

  return im_overlay.astype(image.dtype)