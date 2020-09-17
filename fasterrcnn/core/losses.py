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

"""Classification and regression loss functions for object detection.

Localization losses:
 * WeightedL2LocalizationLoss
 * WeightedSmoothL1LocalizationLoss
 * WeightedIOULocalizationLoss

Classification losses:
 * WeightedSigmoidClassificationLoss
 * WeightedSoftmaxClassificationLoss
 * WeightedSoftmaxClassificationAgainstLogitsLoss
 * BootstrappedSigmoidClassificationLoss
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import tensorflow as tf

class Loss(six.with_metaclass(abc.ABCMeta, object)):
  """Abstract base class for loss functions."""

  def __call__(self,
               prediction_tensor,
               target_tensor,
               ignore_nan_targets=False,
               losses_mask=None,
               scope=None,
               **params):
    """Call the loss function.

    Args:
      prediction_tensor: an N-d tensor of shape [batch, anchors, ...]
        representing predicted quantities.
      target_tensor: an N-d tensor of shape [batch, anchors, ...] representing
        regression or classification targets.
      ignore_nan_targets: whether to ignore nan targets in the loss computation.
        E.g. can be used if the target tensor is missing groundtruth data that
        shouldn't be factored into the loss.
      losses_mask: A [batch] boolean tensor that indicates whether losses should
        be applied to individual images in the batch. For elements that
        are False, corresponding prediction, target, and weight tensors will not
        contribute to loss computation. If None, no filtering will take place
        prior to loss computation.
      scope: Op scope name. Defaults to 'Loss' if None.
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: a tensor representing the value of the loss function.
    """
    with tf.name_scope(scope, 'Loss',
                       [prediction_tensor, target_tensor, params]) as scope:
      if ignore_nan_targets:
        target_tensor = tf.where(tf.is_nan(target_tensor),
                                 prediction_tensor,
                                 target_tensor)
      if losses_mask is not None:
        tensor_multiplier = self._get_loss_multiplier_for_tensor(
            prediction_tensor,
            losses_mask)
        prediction_tensor *= tensor_multiplier
        target_tensor *= tensor_multiplier

        if 'weights' in params:
          params['weights'] = tf.convert_to_tensor(params['weights'])
          weights_multiplier = self._get_loss_multiplier_for_tensor(
              params['weights'],
              losses_mask)
          params['weights'] *= weights_multiplier
      return self._compute_loss(prediction_tensor, target_tensor, **params)

  def _get_loss_multiplier_for_tensor(self, tensor, losses_mask):
    loss_multiplier_shape = tf.stack([-1] + [1] * (len(tensor.shape) - 1))
    return tf.cast(tf.reshape(losses_mask, loss_multiplier_shape), tf.float32)

  @abc.abstractmethod
  def _compute_loss(self, prediction_tensor, target_tensor, **params):
    """Method to be overridden by implementations.

    Args:
      prediction_tensor: a tensor representing predicted quantities
      target_tensor: a tensor representing regression or classification targets
      **params: Additional keyword arguments for specific implementations of
              the Loss.

    Returns:
      loss: an N-d tensor of shape [batch, anchors, ...] containing the loss per
        anchor
    """
    pass

class WeightedSmoothL1LocalizationLoss(Loss):
  """Smooth L1 localization loss function aka Huber Loss..

  The smooth L1_loss is defined elementwise as .5 x^2 if |x| <= delta and
  delta * (|x|- 0.5*delta) otherwise, where x is the difference between
  predictions and target.

  See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
  """

  def __init__(self, delta=1.0):
    """Constructor.

    Args:
      delta: delta for smooth L1 loss.
    """
    super(WeightedSmoothL1LocalizationLoss, self).__init__()
    self._delta = delta

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the (encoded) predicted locations of objects.
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        code_size] representing the regression targets
      weights: a float tensor of shape [batch_size, num_anchors]

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors] tensor
        representing the value of the loss function.
    """
    return tf.reduce_sum(tf.losses.huber_loss(
        target_tensor,
        prediction_tensor,
        delta=self._delta,
        weights=tf.expand_dims(weights, axis=2),
        loss_collection=None,
        reduction=tf.losses.Reduction.NONE
    ), axis=2)


class WeightedSoftmaxClassificationLoss(Loss):
  """Softmax loss function."""

  def __init__(self, logit_scale=1.0):
    """Constructor.

    Args:
      logit_scale: When this value is high, the prediction is "diffused" and
                   when this value is low, the prediction is made peakier.
                   (default 1.0)

    """
    super(WeightedSoftmaxClassificationLoss, self).__init__()
    self._logit_scale = logit_scale

  def _compute_loss(self, prediction_tensor, target_tensor, weights):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape, either [batch_size, num_anchors,
        num_classes] or [batch_size, num_anchors, 1]. If the shape is
        [batch_size, num_anchors, 1], all the classses are equally weighted.

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors]
        representing the value of the loss function.
    """
    weights = tf.reduce_mean(weights, axis=2)
    num_classes = prediction_tensor.get_shape().as_list()[-1]
    prediction_tensor = tf.divide(
        prediction_tensor, self._logit_scale, name='scale_logit')
    per_row_cross_ent = (tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(target_tensor, [-1, num_classes]),
        logits=tf.reshape(prediction_tensor, [-1, num_classes])))
    return tf.reshape(per_row_cross_ent, tf.shape(weights)) * weights

class WeightedSigmoidClassificationLoss(Loss):
  """Sigmoid cross entropy classification loss function."""

  def _compute_loss(self,
                    prediction_tensor,
                    target_tensor,
                    weights,
                    class_indices=None):
    """Compute loss function.

    Args:
      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
      target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
      weights: a float tensor of shape, either [batch_size, num_anchors,
        num_classes] or [batch_size, num_anchors, 1]. If the shape is
        [batch_size, num_anchors, 1], all the classses are equally weighted.
      class_indices: (Optional) A 1-D integer tensor of class indices.
        If provided, computes loss only for the specified class indices.

    Returns:
      loss: a float tensor of shape [batch_size, num_anchors, num_classes]
        representing the value of the loss function.
    """
    if class_indices is not None:
      weights *= tf.reshape(
          ops.indices_to_dense_vector(class_indices,
                                      tf.shape(prediction_tensor)[2]),
          [1, 1, -1])
    per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor))
    return per_entry_cross_ent * weights

