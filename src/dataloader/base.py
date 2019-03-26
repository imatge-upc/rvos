# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# ----------------------------------------------------------------------------

import functools
import os.path as osp

import numpy as np

from PIL import Image
from skimage.io import ImageCollection

from misc.config import cfg, phase
from misc.io_aux import imread_indexed,imwrite_indexed

#################################
# HELPER FUNCTIONS
#################################

def _load_annotation(filename,single_object):
  """ Load image given filename."""

  annotation,_ = imread_indexed(filename)

  if single_object:
    annotation = (annotation != 0).astype(np.uint8)

  return annotation

def _get_num_objects(annotation):
  """ Count number of objects from segmentation mask"""

  ids = sorted(np.unique(annotation))

  # Remove unknown-label
  ids = ids[:-1] if ids[-1] == 255 else ids

  # Handle no-background case
  ids = ids if ids[0] else ids[1:]

  return len(ids)

#################################
# LOADER CLASSES
#################################

class BaseLoader(ImageCollection):

  """
  Base class to load image sets (inherit from skimage.ImageCollection).

  Arguments:
    path      (string): path to sequence folder.
    regex     (string): regular expression to define image search pattern.
    load_func (func)  : function to load image from disk (see skimage.ImageCollection).

  """

  def __init__(self,split,path,regex,load_func=None, lmdb_env=None):
    
    if not lmdb_env == None:
        key_db = osp.basename(path)
        with lmdb_env.begin() as txn:
            _files_vec = txn.get(key_db.encode()).decode().split('|')
            _files = [bytes(osp.join(path, f).encode()) for f in _files_vec]
        super(BaseLoader, self).__init__(_files, load_func=load_func)
    else:  
        super(BaseLoader, self).__init__(
            osp.join(path + '/' + regex),load_func=load_func)

    # Sequence name
    self.name = osp.basename(path)
    self.split = split

  def __str__(self):
    return "< class: '{}' name: '{}', frames: {} >".format(
        type(self).__name__,self.name,len(self))

class Sequence(BaseLoader):

  """
  Load image sequences.

  Arguments:
    name  (string): sequence name.
    regex (string): regular expression to define image search pattern.

  """

  def __init__(self,split,name,regex="*.jpg", lmdb_env=None):
        
    super(Sequence, self).__init__(
        split,osp.join(cfg.PATH.SEQUENCES,name),regex, lmdb_env=lmdb_env)



class SequenceClip_simple:
    """
    Load image sequences.

    Arguments:
      name  (string): sequence name.
      regex (string): regular expression to define image search pattern.

    """

    def __init__(self, seq, starting_frame):

        self.__dict__.update(seq.__dict__)
        self.starting_frame = starting_frame

    def __str__(self):
        return "< class: '{}' name: '{}', startingframe: {}, frames: {} >".format(
            type(self).__name__, self.name, self.starting_frame, len(self))

class SequenceClip(BaseLoader):

  """
  Load image sequences.

  Arguments:
    name  (string): sequence name.
    regex (string): regular expression to define image search pattern.

  """

  def __init__(self,split,name,starting_frame,regex="*.jpg", lmdb_env=None):
    
    super(SequenceClip, self).__init__(
        split,osp.join(cfg.PATH.SEQUENCES,name),regex, lmdb_env=lmdb_env)

    self.starting_frame = starting_frame
    
  def __str__(self):
    return "< class: '{}' name: '{}', startingframe: {}, frames: {} >".format(
        type(self).__name__,self.name,self.starting_frame,len(self))

class Segmentation(BaseLoader):

  """
  Load image sequences.

  Arguments:
    path          (string): path to sequence folder.
    single_object (bool):   assign same id=1 to each object.
    regex         (string): regular expression to define image search pattern.

  """

  def __init__(self,split,path,single_object,regex="*.png", lmdb_env=None):
    super(Segmentation, self).__init__(split,path,regex,
       functools.partial(_load_annotation,single_object=single_object), lmdb_env=lmdb_env)

    self.n_objects = _get_num_objects(self[0])

  def iter_objects_id(self):
    """
    Iterate over objects providing object id for each of them.
    """
    for obj_id in range(1,self.n_objects+1):
      yield obj_id

  def iter_objects(self):
    """
    Iterate over objects providing binary masks for each of them.
    """

    for obj_id in self.iter_objects_id():
      bn_segmentation = [(s==obj_id).astype(np.uint8) for s in self]
      yield bn_segmentation

class Annotation(Segmentation):

  """
  Load ground-truth annotations.

  Arguments:
    name          (string): sequence name.
    single_object (bool):   assign same id=1 to each object.
    regex         (string): regular expression to define image search pattern.

  """

  def __init__(self,split,name,single_object,regex="*.png", lmdb_env=None):
    
    super(Annotation, self).__init__(
        split,osp.join(cfg.PATH.ANNOTATIONS,name),single_object,regex, lmdb_env=lmdb_env)


class AnnotationClip_simple:
    """
    Load ground-truth annotations.

    Arguments:
      name          (string): sequence name.
      single_object (bool):   assign same id=1 to each object.
      regex         (string): regular expression to define image search pattern.

    """

    def __init__(self, annot, starting_frame):

        self.__dict__.update(annot.__dict__)
        self.starting_frame = starting_frame

class AnnotationClip(Segmentation):

  """
  Load ground-truth annotations.

  Arguments:
    name          (string): sequence name.
    single_object (bool):   assign same id=1 to each object.
    regex         (string): regular expression to define image search pattern.

  """

  def __init__(self,split,name,starting_frame,single_object,regex="*.png", lmdb_env=None):

    super(AnnotationClip, self).__init__(
        split,osp.join(cfg.PATH.ANNOTATIONS,name),single_object,regex, lmdb_env=lmdb_env)
    self.starting_frame = starting_frame
    