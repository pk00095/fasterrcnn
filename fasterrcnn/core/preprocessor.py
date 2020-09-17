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

def resize_to_range(image,
                    masks=None,
                    min_dimension=None,
                    max_dimension=None,
                    method=tf.image.ResizeMethod.BILINEAR,
                    align_corners=False,
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
    return tf.image.resize_images(
        image, tf.stack([min_dimension, max_dimension]), method=method,
        align_corners=align_corners, preserve_aspect_ratio=True)

  def _resize_portrait_image(image):
    # resize a portrait image
    return tf.image.resize_images(
        image, tf.stack([max_dimension, min_dimension]), method=method,
        align_corners=align_corners, preserve_aspect_ratio=True)

  with tf.name_scope('ResizeToRange', values=[image, min_dimension]):
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
    if masks is not None:
      new_masks = tf.expand_dims(masks, 3)
      new_masks = tf.image.resize_images(
          new_masks,
          new_size[:-1],
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
          align_corners=align_corners)
      if pad_to_max_dimension:
        new_masks = tf.image.pad_to_bounding_box(
            new_masks, 0, 0, max_dimension, max_dimension)
      new_masks = tf.squeeze(new_masks, 3)
      result.append(new_masks)

    result.append(new_size)
    return result


