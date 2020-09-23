from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import range
import tensorflow as tf

from fasterrcnn.core import box_list

def gather(boxlist, indices, fields=None, scope=None, use_static_shapes=False):
    """Gather boxes from BoxList according to indices and return new BoxList.

    By default, `gather` returns boxes corresponding to the input index list, as
    well as all additional fields stored in the boxlist (indexing into the
    first dimension).  However one can optionally only gather from a
    subset of fields.

    Args:
      boxlist: BoxList holding N boxes
      indices: a rank-1 tensor of type int32 / int64
      fields: (optional) list of fields to also gather from.  If None (default),
        all fields are gathered from.  Pass an empty fields list to only gather
        the box coordinates.
      scope: name scope.
      use_static_shapes: Whether to use an implementation with static shape
        gurantees.

    Returns:
      subboxlist: a BoxList corresponding to the subset of the input BoxList
      specified by indices
    Raises:
      ValueError: if specified field is not contained in boxlist or if the
        indices are not of type int32
    """
    # with tf.name_scope(scope, 'Gather'):
    if len(indices.shape.as_list()) != 1:
      raise ValueError('indices should have rank 1')
    if indices.dtype != tf.int32 and indices.dtype != tf.int64:
      raise ValueError('indices should be an int32 / int64 tensor')
    gather_op = tf.gather
    if use_static_shapes:
      gather_op = ops.matmul_gather_on_zeroth_axis
    subboxlist = box_list.BoxList(gather_op(boxlist.get(), indices))
    if fields is None:
      fields = boxlist.get_extra_fields()
    fields += ['boxes']
    for field in fields:
      if not boxlist.has_field(field):
        raise ValueError('boxlist must contain all specified fields')
      subfieldlist = gather_op(boxlist.get_field(field), indices)
      subboxlist.add_field(field, subfieldlist)
    return subboxlist


def height_width(boxlist, scope=None):
    """Computes height and width of boxes in boxlist.

    Args:
      boxlist: BoxList holding N boxes
      scope: name scope.

    Returns:
      Height: A tensor with shape [N] representing box heights.
      Width: A tensor with shape [N] representing box widths.
    """
    # with tf.name_scope(scope, 'HeightWidth'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    return tf.squeeze(y_max - y_min, [1]), tf.squeeze(x_max - x_min, [1])

def prune_small_boxes(boxlist, min_side, scope=None):
    """Prunes small boxes in the boxlist which have a side smaller than min_side.

    Args:
      boxlist: BoxList holding N boxes.
      min_side: Minimum width AND height of box to survive pruning.
      scope: name scope.

    Returns:
      A pruned boxlist.
    """
    # with tf.name_scope(scope, 'PruneSmallBoxes'):
    height, width = height_width(boxlist)
    is_valid = tf.logical_and(tf.greater_equal(width, min_side),
                              tf.greater_equal(height, min_side))
    return gather(boxlist, tf.reshape(tf.where(is_valid), [-1]))


def area(boxlist, scope=None):
  """Computes area of boxes.

  Args:
    boxlist: BoxList holding N boxes
    scope: name scope.

  Returns:
    a tensor with shape [N] representing box areas.
  """
  with tf.name_scope(scope, 'Area'):
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min), [1])

def intersection(boxlist1, boxlist2, scope=None):
  """Compute pairwise intersection areas between boxes.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise intersections
  """
  with tf.name_scope(scope, 'Intersection'):
    y_min1, x_min1, y_max1, x_max1 = tf.split(
        value=boxlist1.get(), num_or_size_splits=4, axis=1)
    y_min2, x_min2, y_max2, x_max2 = tf.split(
        value=boxlist2.get(), num_or_size_splits=4, axis=1)
    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2))
    intersect_heights = tf.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths

def iou(boxlist1, boxlist2, scope=None):
  """Computes pairwise intersection-over-union between box collections.

  Args:
    boxlist1: BoxList holding N boxes
    boxlist2: BoxList holding M boxes
    scope: name scope.

  Returns:
    a tensor with shape [N, M] representing pairwise iou scores.
  """
  with tf.name_scope(scope, 'IOU'):
    intersections = intersection(boxlist1, boxlist2)
    areas1 = area(boxlist1)
    areas2 = area(boxlist2)
    unions = (
        tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
    return tf.where(
        tf.equal(intersections, 0.0),
        tf.zeros_like(intersections), tf.truediv(intersections, unions))

def _copy_extra_fields(boxlist_to_copy_to, boxlist_to_copy_from):
  """Copies the extra fields of boxlist_to_copy_from to boxlist_to_copy_to.

  Args:
    boxlist_to_copy_to: BoxList to which extra fields are copied.
    boxlist_to_copy_from: BoxList from which fields are copied.

  Returns:
    boxlist_to_copy_to with extra fields.
  """
  for field in boxlist_to_copy_from.get_extra_fields():
    boxlist_to_copy_to.add_field(field, boxlist_to_copy_from.get_field(field))
  return boxlist_to_copy_to

def scale(boxlist, y_scale, x_scale, scope=None):
    """scale box coordinates in x and y dimensions.

    Args:
      boxlist: BoxList holding N boxes
      y_scale: (float) scalar tensor
      x_scale: (float) scalar tensor
      scope: name scope.

    Returns:
      boxlist: BoxList holding N boxes
    """
    # with tf.name_scope(scope, 'Scale'):
    y_scale = tf.cast(y_scale, tf.float32)
    x_scale = tf.cast(x_scale, tf.float32)
    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    y_min = y_scale * y_min
    y_max = y_scale * y_max
    x_min = x_scale * x_min
    x_max = x_scale * x_max
    scaled_boxlist = box_list.BoxList(
        tf.concat([y_min, x_min, y_max, x_max], 1))
    return _copy_extra_fields(scaled_boxlist, boxlist)


def change_coordinate_frame(boxlist, window, scope=None):
  """Change coordinate frame of the boxlist to be relative to window's frame.

  Given a window of the form [ymin, xmin, ymax, xmax],
  changes bounding box coordinates from boxlist to be relative to this window
  (e.g., the min corner maps to (0,0) and the max corner maps to (1,1)).

  An example use case is data augmentation: where we are given groundtruth
  boxes (boxlist) and would like to randomly crop the image to some
  window (window). In this case we need to change the coordinate frame of
  each groundtruth box to be relative to this new window.

  Args:
    boxlist: A BoxList object holding N boxes.
    window: A rank 1 tensor [4].
    scope: name scope.

  Returns:
    Returns a BoxList object with N boxes.
  """
  # with tf.name_scope(scope, 'ChangeCoordinateFrame'):
  win_height = window[2] - window[0]
  win_width = window[3] - window[1]
  boxlist_new = scale(box_list.BoxList(
      boxlist.get() - [window[0], window[1], window[0], window[1]]),
                      1.0 / win_height, 1.0 / win_width)
  boxlist_new = _copy_extra_fields(boxlist_new, boxlist)
  return boxlist_new