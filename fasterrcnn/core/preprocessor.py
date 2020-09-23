# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Preprocess images and bounding boxes for detection.

We perform two sets of operations in preprocessing stage:
(a) operations that are applied to both training and testing data,
(b) operations that are applied only to training data for the purpose of
    data augmentation.

A preprocessing function receives a set of inputs,
e.g. an image and bounding boxes,
performs an operation on them, and returns them.
Some examples are: randomly cropping the image, randomly mirroring the image,
                   randomly changing the brightness, contrast, hue and
                   randomly jittering the bounding boxes.

The preprocess function receives a tensor_dict which is a dictionary that maps
different field names to their tensors. For example,
tensor_dict[fields.InputDataFields.image] holds the image tensor.
The image is a rank 4 tensor: [1, height, width, channels] with
dtype=tf.float32. The groundtruth_boxes is a rank 2 tensor: [N, 4] where
in each row there is a box with [ymin xmin ymax xmax].
Boxes are in normalized coordinates meaning
their coordinate values range in [0, 1]

To preprocess multiple images with the same operations in cases where
nondeterministic operations are used, a preprocessor_cache.PreprocessorCache
object can be passed into the preprocess function or individual operations.
All nondeterministic operations except random_jitter_boxes support caching.
E.g.
Let tensor_dict{1,2,3,4,5} be copies of the same inputs.
Let preprocess_options contain nondeterministic operation(s) excluding
random_jitter_boxes.

cache1 = preprocessor_cache.PreprocessorCache()
cache2 = preprocessor_cache.PreprocessorCache()
a = preprocess(tensor_dict1, preprocess_options, preprocess_vars_cache=cache1)
b = preprocess(tensor_dict2, preprocess_options, preprocess_vars_cache=cache1)
c = preprocess(tensor_dict3, preprocess_options, preprocess_vars_cache=cache2)
d = preprocess(tensor_dict4, preprocess_options, preprocess_vars_cache=cache2)
e = preprocess(tensor_dict5, preprocess_options)

Then correspondings tensors of object pairs (a,b) and (c,d)
are guaranteed to be equal element-wise, but the equality of any other object
pair cannot be determined.

Important Note: In tensor_dict, images is a rank 4 tensor, but preprocessing
functions receive a rank 3 tensor for processing the image. Thus, inside the
preprocess function we squeeze the image to become a rank 3 tensor and then
we pass it to the functions. At the end of the preprocess we expand the image
back to rank 4.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import sys

import six
from six.moves import range
from six.moves import zip
import tensorflow as tf

from fasterrcnn.core import standard_fields as fields


def normalize_image(image, original_minval, original_maxval, target_minval,
                    target_maxval):
  """Normalizes pixel values in the image.

  Moves the pixel values from the current [original_minval, original_maxval]
  range to a the [target_minval, target_maxval] range.

  Args:
    image: rank 3 float32 tensor containing 1
           image -> [height, width, channels].
    original_minval: current image minimum value.
    original_maxval: current image maximum value.
    target_minval: target image minimum value.
    target_maxval: target image maximum value.

  Returns:
    image: image which is the same shape as input image.
  """
  with tf.name_scope('NormalizeImage', values=[image]):
    original_minval = float(original_minval)
    original_maxval = float(original_maxval)
    target_minval = float(target_minval)
    target_maxval = float(target_maxval)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.subtract(image, original_minval)
    image = tf.multiply(image, (target_maxval - target_minval) /
                        (original_maxval - original_minval))
    image = tf.add(image, target_minval)
    return image



def _flip_boxes_left_right(boxes):
  """Left-right flip the boxes.

  Args:
    boxes: Float32 tensor containing the bounding boxes -> [..., 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each last dimension is in the form of [ymin, xmin, ymax, xmax].

  Returns:
    Flipped boxes.
  """
  ymin, xmin, ymax, xmax = tf.split(value=boxes, num_or_size_splits=4, axis=-1)
  flipped_xmin = tf.subtract(1.0, xmax)
  flipped_xmax = tf.subtract(1.0, xmin)
  flipped_boxes = tf.concat([ymin, flipped_xmin, ymax, flipped_xmax], axis=-1)
  return flipped_boxes

@tf.function
def resize_to_range(image,
                    min_dimension=None,
                    max_dimension=None,
                    method=tf.image.ResizeMethod.BILINEAR,
                    # align_corners=False,
                    pad_to_max_dimension=False,
                    per_channel_pad_value=(0, 0, 0)):
  """Resizes an image so its dimensions are within the provided value.

  The output size can be described by two cases:
  1. If the image can be rescaled so its minimum dimension is equal to the
     provided value without the other dimension exceeding max_dimension,
     then do so.
  2. Otherwise, resize so the largest dimension is equal to max_dimension.

  Args:
    image: A 3D tensor of shape [height, width, channels]
    masks: (optional) rank 3 float32 tensor with shape
           [num_instances, height, width] containing instance masks.
    min_dimension: (optional) (scalar) desired size of the smaller image
                   dimension.
    max_dimension: (optional) (scalar) maximum allowed size
                   of the larger image dimension.
    method: (optional) interpolation method used in resizing. Defaults to
            BILINEAR.
    align_corners: bool. If true, exactly align all 4 corners of the input
                   and output. Defaults to False.
    pad_to_max_dimension: Whether to resize the image and pad it with zeros
      so the resulting image is of the spatial size
      [max_dimension, max_dimension]. If masks are included they are padded
      similarly.
    per_channel_pad_value: A tuple of per-channel scalar value to use for
      padding. By default pads zeros.

  Returns:
    Note that the position of the resized_image_shape changes based on whether
    masks are present.
    resized_image: A 3D tensor of shape [new_height, new_width, channels],
      where the image has been resized (with bilinear interpolation) so that
      min(new_height, new_width) == min_dimension or
      max(new_height, new_width) == max_dimension.
    resized_masks: If masks is not None, also outputs masks. A 3D tensor of
      shape [num_instances, new_height, new_width].
    resized_image_shape: A 1D tensor of shape [3] containing shape of the
      resized image.

  Raises:
    ValueError: if the image is not a 3D tensor.
  """
  if len(image.get_shape()) != 3:
    raise ValueError('Image should be 3D tensor')

  def _resize_landscape_image(image):
    # resize a landscape image
    return tf.image.resize(
        image, tf.stack([min_dimension, max_dimension]), method=method, preserve_aspect_ratio=True)

  def _resize_portrait_image(image):
    # resize a portrait image
    return tf.image.resize(
        image, tf.stack([max_dimension, min_dimension]), method=method, preserve_aspect_ratio=True)

  # with tf.name_scope('ResizeToRange', values=[image, min_dimension]):
  if image.get_shape().is_fully_defined():
    if image.get_shape()[0] < image.get_shape()[1]:
      new_image = _resize_landscape_image(image)
    else:
      new_image = _resize_portrait_image(image)
    new_size = tf.constant(new_image.get_shape().as_list())
  else:
    new_image = tf.cond(
        tf.less(tf.shape(image)[0], tf.shape(image)[1]),
        lambda: _resize_landscape_image(image),
        lambda: _resize_portrait_image(image))
    new_size = tf.shape(new_image)

  if pad_to_max_dimension:
    channels = tf.unstack(new_image, axis=2)
    if len(channels) != len(per_channel_pad_value):
      raise ValueError('Number of channels must be equal to the length of '
                       'per-channel pad value.')
    new_image = tf.stack(
        [
            tf.pad(
                channels[i], [[0, max_dimension - new_size[0]],
                              [0, max_dimension - new_size[1]]],
                constant_values=per_channel_pad_value[i])
            for i in range(len(channels))
        ],
        axis=2)
    new_image.set_shape([max_dimension, max_dimension, 3])

    result = [new_image]
    result.append(new_size)
    return result

def _get_or_create_preprocess_rand_vars(generator_func,
                                        function_id,
                                        preprocess_vars_cache,
                                        key=''):
  """Returns a tensor stored in preprocess_vars_cache or using generator_func.

  If the tensor was previously generated and appears in the PreprocessorCache,
  the previously generated tensor will be returned. Otherwise, a new tensor
  is generated using generator_func and stored in the cache.

  Args:
    generator_func: A 0-argument function that generates a tensor.
    function_id: identifier for the preprocessing function used.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.
    key: identifier for the variable stored.
  Returns:
    The generated tensor.
  """
  if preprocess_vars_cache is not None:
    var = preprocess_vars_cache.get(function_id, key)
    if var is None:
      var = generator_func()
      preprocess_vars_cache.update(function_id, key, var)
  else:
    var = generator_func()
  return var


def random_horizontal_flip(image,
                           boxes,
                           # masks=None,
                           # keypoints=None,
                           # keypoint_visibilities=None,
                           # densepose_part_ids=None,
                           # densepose_surface_coords=None,
                           # keypoint_flip_permutation=None,
                           # probability=0.5,
                           seed=None,
                           preprocess_vars_cache=None):
  """Randomly flips the image and detections horizontally.

  Args:
    image: rank 3 float32 tensor with shape [height, width, channels].
    boxes: rank 2 float32 tensor with shape [N, 4]
           containing the bounding boxes.
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
           Each row is in the form of [ymin, xmin, ymax, xmax].
    probability: the probability of performing this augmentation.
    seed: random seed
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.

  Returns:
    image: image which is the same shape as input image.

    If boxes, masks, keypoints, keypoint_visibilities,
    keypoint_flip_permutation, densepose_part_ids, or densepose_surface_coords
    are not None,the function also returns the following tensors.

    boxes: rank 2 float32 tensor containing the bounding boxes -> [N, 4].
           Boxes are in normalized form meaning their coordinates vary
           between [0, 1].
    masks: rank 3 float32 tensor with shape [num_instances, height, width]
           containing instance masks.
    keypoints: rank 3 float32 tensor with shape
               [num_instances, num_keypoints, 2]
    keypoint_visibilities: rank 2 bool tensor with shape
                           [num_instances, num_keypoints].
    densepose_part_ids: rank 2 int32 tensor with shape
                        [num_instances, num_points].
    densepose_surface_coords: rank 3 float32 tensor with shape
                              [num_instances, num_points, 4].

  Raises:
    ValueError: if keypoints are provided but keypoint_flip_permutation is not.
    ValueError: if either densepose_part_ids or densepose_surface_coords is
                not None, but both are not None.
  """

  def _flip_image(image):
    # flip image
    image_flipped = tf.image.flip_left_right(image)
    return image_flipped

  # if keypoints is not None and keypoint_flip_permutation is None:
  #   raise ValueError(
  #       'keypoints are provided but keypoints_flip_permutation is not provided')

  # if ((densepose_part_ids is not None and densepose_surface_coords is None) or
  #     (densepose_part_ids is None and densepose_surface_coords is not None)):
  #   raise ValueError(
  #       'Must provide both `densepose_part_ids` and `densepose_surface_coords`')

  # with tf.name_scope('RandomHorizontalFlip', values=[image, boxes]):
  result = []
  # random variable defining whether to do flip or not
  # generator_func = functools.partial(tf.random.uniform, [], seed=seed)
  # do_a_flip_random = _get_or_create_preprocess_rand_vars(
  #     generator_func,
  #     1,
  #     None)
  probability=0.5
  do_a_flip_random = tf.less(tf.random.uniform(shape=[], seed=seed), probability)

  # flip image
  image = tf.cond(do_a_flip_random, lambda: _flip_image(image), lambda: image)
  result.append(image)

  # flip boxes
  # if boxes is not None:
  boxes = tf.cond(do_a_flip_random, lambda: _flip_boxes_left_right(boxes),
                  lambda: boxes)
  result.append(boxes)
  return tuple(result)

def get_default_func_arg_map(include_label_weights=True,
                             include_label_confidences=False,
                             include_multiclass_scores=False,
                             include_instance_masks=False,
                             include_keypoints=False,
                             include_keypoint_visibilities=False,
                             include_dense_pose=False):
  """Returns the default mapping from a preprocessor function to its args.

  Args:
    include_label_weights: If True, preprocessing functions will modify the
      label weights, too.
    include_label_confidences: If True, preprocessing functions will modify the
      label confidences, too.
    include_multiclass_scores: If True, preprocessing functions will modify the
      multiclass scores, too.
    include_instance_masks: If True, preprocessing functions will modify the
      instance masks, too.
    include_keypoints: If True, preprocessing functions will modify the
      keypoints, too.
    include_keypoint_visibilities: If True, preprocessing functions will modify
      the keypoint visibilities, too.
    include_dense_pose: If True, preprocessing functions will modify the
      DensePose labels, too.

  Returns:
    A map from preprocessing functions to the arguments they receive.
  """
  groundtruth_label_weights = None
  if include_label_weights:
    groundtruth_label_weights = (
        fields.InputDataFields.groundtruth_weights)

  groundtruth_label_confidences = None

  multiclass_scores = None

  groundtruth_instance_masks = None

  groundtruth_keypoints = None

  groundtruth_keypoint_visibilities = None

  groundtruth_dp_num_points = None
  groundtruth_dp_part_ids = None
  groundtruth_dp_surface_coords = None

  prep_func_arg_map = {
      # normalize_image: (fields.InputDataFields.image,),
      random_horizontal_flip: (
          fields.InputDataFields.image,
          fields.InputDataFields.groundtruth_boxes,
          # groundtruth_instance_masks,
          # groundtruth_keypoints,
          # groundtruth_keypoint_visibilities,
          groundtruth_dp_part_ids,
          groundtruth_dp_surface_coords,
      ),
  }

  return prep_func_arg_map


def preprocess(tensor_dict,
               preprocess_options,
               func_arg_map=None,
               preprocess_vars_cache=None):
  """Preprocess images and bounding boxes.

  Various types of preprocessing (to be implemented) based on the
  preprocess_options dictionary e.g. "crop image" (affects image and possibly
  boxes), "white balance image" (affects only image), etc. If self._options
  is None, no preprocessing is done.

  Args:
    tensor_dict: dictionary that contains images, boxes, and can contain other
                 things as well.
                 images-> rank 4 float32 tensor contains
                          1 image -> [1, height, width, 3].
                          with pixel values varying between [0, 1]
                 boxes-> rank 2 float32 tensor containing
                         the bounding boxes -> [N, 4].
                         Boxes are in normalized form meaning
                         their coordinates vary between [0, 1].
                         Each row is in the form
                         of [ymin, xmin, ymax, xmax].
    preprocess_options: It is a list of tuples, where each tuple contains a
                        function and a dictionary that contains arguments and
                        their values.
    func_arg_map: mapping from preprocessing functions to arguments that they
                  expect to receive and return.
    preprocess_vars_cache: PreprocessorCache object that records previously
                           performed augmentations. Updated in-place. If this
                           function is called multiple times with the same
                           non-null cache, it will perform deterministically.

  Returns:
    tensor_dict: which contains the preprocessed images, bounding boxes, etc.

  Raises:
    ValueError: (a) If the functions passed to Preprocess
                    are not in func_arg_map.
                (b) If the arguments that a function needs
                    do not exist in tensor_dict.
                (c) If image in tensor_dict is not rank 4
  """
  if func_arg_map is None:
    func_arg_map = get_default_func_arg_map()
  # changes the images to image (rank 4 to rank 3) since the functions
  # receive rank 3 tensor for image
  if fields.InputDataFields.image in tensor_dict:
    images = tensor_dict[fields.InputDataFields.image]
    if len(images.get_shape()) != 4:
      raise ValueError('images in tensor_dict should be rank 4')
    image = tf.squeeze(images, axis=0)
    tensor_dict[fields.InputDataFields.image] = image

  # Preprocess inputs based on preprocess_options
  for option in preprocess_options:
    func, params = option
    # print(params)
    if func not in func_arg_map:
      raise ValueError('The function %s does not exist in func_arg_map' %
                       (func.__name__))
    arg_names = func_arg_map[func]
    for a in arg_names:
      if a is not None and a not in tensor_dict:
        raise ValueError('The function %s requires argument %s' %
                         (func.__name__, a))

    def get_arg(key):
      return tensor_dict[key] if key is not None else None

    args = [get_arg(a) for a in arg_names]
    if preprocess_vars_cache is not None:
      if six.PY2:
        # pylint: disable=deprecated-method
        arg_spec = inspect.getargspec(func)
        # pylint: enable=deprecated-method
      else:
        arg_spec = inspect.getfullargspec(func)
      if 'preprocess_vars_cache' in arg_spec.args:
        params['preprocess_vars_cache'] = preprocess_vars_cache

    results = func(*args, **params)
    if not isinstance(results, (list, tuple)):
      results = (results,)
    # Removes None args since the return values will not contain those.
    arg_names = [arg_name for arg_name in arg_names if arg_name is not None]
    for res, arg_name in zip(results, arg_names):
      tensor_dict[arg_name] = res

  # changes the image to images (rank 3 to rank 4) to be compatible to what
  # we received in the first place
  if fields.InputDataFields.image in tensor_dict:
    image = tensor_dict[fields.InputDataFields.image]
    images = tf.expand_dims(image, 0)
    tensor_dict[fields.InputDataFields.image] = images

  return tensor_dict
