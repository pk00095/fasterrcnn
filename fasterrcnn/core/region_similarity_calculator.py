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

"""Region Similarity Calculators for BoxLists.

Region Similarity Calculators compare a pairwise measure of similarity
between the boxes in two BoxLists.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod

import six
import tensorflow as tf

from object_detection.core import box_list_ops

class RegionSimilarityCalculator(six.with_metaclass(ABCMeta, object)):
  """Abstract base class for region similarity calculator."""

  def compare(self, boxlist1, boxlist2, scope=None):
    """Computes matrix of pairwise similarity between BoxLists.

    This op (to be overridden) computes a measure of pairwise similarity between
    the boxes in the given BoxLists. Higher values indicate more similarity.

    Note that this method simply measures similarity and does not explicitly
    perform a matching.

    Args:
      boxlist1: BoxList holding N boxes.
      boxlist2: BoxList holding M boxes.
      scope: Op scope name. Defaults to 'Compare' if None.

    Returns:
      a (float32) tensor of shape [N, M] with pairwise similarity score.
    """
    with tf.name_scope(scope, 'Compare', [boxlist1, boxlist2]) as scope:
      return self._compare(boxlist1, boxlist2)

  @abstractmethod
  def _compare(self, boxlist1, boxlist2):
    pass


class IouSimilarity(RegionSimilarityCalculator):
  """Class to compute similarity based on Intersection over Union (IOU) metric.

  This class computes pairwise similarity between two BoxLists based on IOU.
  """

  def _compare(self, boxlist1, boxlist2):
    """Compute pairwise IOU similarity between the two BoxLists.

    Args:
      boxlist1: BoxList holding N boxes.
      boxlist2: BoxList holding M boxes.

    Returns:
      A tensor with shape [N, M] representing pairwise iou scores.
    """
    return box_list_ops.iou(boxlist1, boxlist2)


