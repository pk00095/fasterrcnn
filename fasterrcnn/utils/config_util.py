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
"""Functions for reading and updating configuration files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from google.protobuf import text_format

from tensorflow.python.lib.io import file_io

from fasterrcnn.protos import pipeline_pb2

def get_configs_from_pipeline_file(pipeline_config_path, config_override=None):
  """Reads config from a file containing pipeline_pb2.TrainEvalPipelineConfig.

  Args:
    pipeline_config_path: Path to pipeline_pb2.TrainEvalPipelineConfig text
      proto.
    config_override: A pipeline_pb2.TrainEvalPipelineConfig text proto to
      override pipeline_config_path.

  Returns:
    Dictionary of configuration objects. Keys are `model`, `train_config`,
      `train_input_config`, `eval_config`, `eval_input_config`. Value are the
      corresponding config objects.
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.io.gfile.GFile(pipeline_config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)
  if config_override:
    text_format.Merge(config_override, pipeline_config)
  return create_configs_from_pipeline_proto(pipeline_config)


def create_configs_from_pipeline_proto(pipeline_config):
  """Creates a configs dictionary from pipeline_pb2.TrainEvalPipelineConfig.

  Args:
    pipeline_config: pipeline_pb2.TrainEvalPipelineConfig proto object.

  Returns:
    Dictionary of configuration objects. Keys are `model`, `train_config`,
      `train_input_config`, `eval_config`, `eval_input_configs`. Value are
      the corresponding config objects or list of config objects (only for
      eval_input_configs).
  """
  configs = {}
  configs["model"] = pipeline_config.model
  configs["train_config"] = pipeline_config.train_config
  configs["train_input_config"] = pipeline_config.train_input_reader
  configs["eval_config"] = pipeline_config.eval_config
  configs["eval_input_configs"] = pipeline_config.eval_input_reader
  # Keeps eval_input_config only for backwards compatibility. All clients should
  # read eval_input_configs instead.
  if configs["eval_input_configs"]:
    configs["eval_input_config"] = configs["eval_input_configs"][0]
  if pipeline_config.HasField("graph_rewriter"):
    configs["graph_rewriter_config"] = pipeline_config.graph_rewriter

  return configs


