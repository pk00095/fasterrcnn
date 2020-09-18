# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: fasterrcnn/protos/faster_rcnn.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from fasterrcnn.protos import anchor_generator_pb2 as fasterrcnn_dot_protos_dot_anchor__generator__pb2
from fasterrcnn.protos import box_predictor_pb2 as fasterrcnn_dot_protos_dot_box__predictor__pb2
from fasterrcnn.protos import hyperparams_pb2 as fasterrcnn_dot_protos_dot_hyperparams__pb2
from fasterrcnn.protos import image_resizer_pb2 as fasterrcnn_dot_protos_dot_image__resizer__pb2
from fasterrcnn.protos import losses_pb2 as fasterrcnn_dot_protos_dot_losses__pb2
from fasterrcnn.protos import post_processing_pb2 as fasterrcnn_dot_protos_dot_post__processing__pb2
from fasterrcnn.protos import fpn_pb2 as fasterrcnn_dot_protos_dot_fpn__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='fasterrcnn/protos/faster_rcnn.proto',
  package='fasterrcnn.protos',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=_b('\n#fasterrcnn/protos/faster_rcnn.proto\x12\x11\x66\x61sterrcnn.protos\x1a(fasterrcnn/protos/anchor_generator.proto\x1a%fasterrcnn/protos/box_predictor.proto\x1a#fasterrcnn/protos/hyperparams.proto\x1a%fasterrcnn/protos/image_resizer.proto\x1a\x1e\x66\x61sterrcnn/protos/losses.proto\x1a\'fasterrcnn/protos/post_processing.proto\x1a\x1b\x66\x61sterrcnn/protos/fpn.proto\"\xb3\x0f\n\nFasterRcnn\x12\x1b\n\x10number_of_stages\x18\x01 \x01(\x05:\x01\x32\x12\x13\n\x0bnum_classes\x18\x03 \x01(\x05\x12\x36\n\rimage_resizer\x18\x04 \x01(\x0b\x32\x1f.fasterrcnn.protos.ImageResizer\x12H\n\x11\x66\x65\x61ture_extractor\x18\x05 \x01(\x0b\x32-.fasterrcnn.protos.FasterRcnnFeatureExtractor\x12H\n\x1c\x66irst_stage_anchor_generator\x18\x06 \x01(\x0b\x32\".fasterrcnn.protos.AnchorGenerator\x12\"\n\x17\x66irst_stage_atrous_rate\x18\x07 \x01(\x05:\x01\x31\x12R\n*first_stage_box_predictor_conv_hyperparams\x18\x08 \x01(\x0b\x32\x1e.fasterrcnn.protos.Hyperparams\x12\x30\n%first_stage_box_predictor_kernel_size\x18\t \x01(\x05:\x01\x33\x12,\n\x1f\x66irst_stage_box_predictor_depth\x18\n \x01(\x05:\x03\x35\x31\x32\x12\'\n\x1a\x66irst_stage_minibatch_size\x18\x0b \x01(\x05:\x03\x32\x35\x36\x12\x32\n%first_stage_positive_balance_fraction\x18\x0c \x01(\x02:\x03\x30.5\x12*\n\x1f\x66irst_stage_nms_score_threshold\x18\r \x01(\x02:\x01\x30\x12*\n\x1d\x66irst_stage_nms_iou_threshold\x18\x0e \x01(\x02:\x03\x30.7\x12&\n\x19\x66irst_stage_max_proposals\x18\x0f \x01(\x05:\x03\x33\x30\x30\x12/\n$first_stage_localization_loss_weight\x18\x10 \x01(\x02:\x01\x31\x12-\n\"first_stage_objectness_loss_weight\x18\x11 \x01(\x02:\x01\x31\x12\x19\n\x11initial_crop_size\x18\x12 \x01(\x05\x12\x1b\n\x13maxpool_kernel_size\x18\x13 \x01(\x05\x12\x16\n\x0emaxpool_stride\x18\x14 \x01(\x05\x12\x43\n\x1asecond_stage_box_predictor\x18\x15 \x01(\x0b\x32\x1f.fasterrcnn.protos.BoxPredictor\x12#\n\x17second_stage_batch_size\x18\x16 \x01(\x05:\x02\x36\x34\x12+\n\x1dsecond_stage_balance_fraction\x18\x17 \x01(\x02:\x04\x30.25\x12G\n\x1csecond_stage_post_processing\x18\x18 \x01(\x0b\x32!.fasterrcnn.protos.PostProcessing\x12\x30\n%second_stage_localization_loss_weight\x18\x19 \x01(\x02:\x01\x31\x12\x32\n\'second_stage_classification_loss_weight\x18\x1a \x01(\x02:\x01\x31\x12\x33\n(second_stage_mask_prediction_loss_weight\x18\x1b \x01(\x02:\x01\x31\x12?\n\x12hard_example_miner\x18\x1c \x01(\x0b\x32#.fasterrcnn.protos.HardExampleMiner\x12O\n second_stage_classification_loss\x18\x1d \x01(\x0b\x32%.fasterrcnn.protos.ClassificationLoss\x12\'\n\x18inplace_batchnorm_update\x18\x1e \x01(\x08:\x05\x66\x61lse\x12)\n\x1ause_matmul_crop_and_resize\x18\x1f \x01(\x08:\x05\x66\x61lse\x12$\n\x15\x63lip_anchors_to_image\x18  \x01(\x08:\x05\x66\x61lse\x12+\n\x1cuse_matmul_gather_in_matcher\x18! \x01(\x08:\x05\x66\x61lse\x12\x30\n!use_static_balanced_label_sampler\x18\" \x01(\x08:\x05\x66\x61lse\x12 \n\x11use_static_shapes\x18# \x01(\x08:\x05\x66\x61lse\x12\x1a\n\x0cresize_masks\x18$ \x01(\x08:\x04true\x12)\n\x1ause_static_shapes_for_eval\x18% \x01(\x08:\x05\x66\x61lse\x12\x30\n\"use_partitioned_nms_in_first_stage\x18& \x01(\x08:\x04true\x12\x33\n$return_raw_detections_during_predict\x18\' \x01(\x08:\x05\x66\x61lse\x12.\n\x1fuse_combined_nms_in_first_stage\x18( \x01(\x08:\x05\x66\x61lse\x12(\n\x19output_final_box_features\x18* \x01(\x08:\x05\x66\x61lse\x12\x32\n\x0e\x63ontext_config\x18) \x01(\x0b\x32\x1a.fasterrcnn.protos.Context\"\xaa\x01\n\x07\x43ontext\x12&\n\x18max_num_context_features\x18\x01 \x01(\x05:\x04\x32\x30\x30\x30\x12,\n\x1e\x61ttention_bottleneck_dimension\x18\x02 \x01(\x05:\x04\x32\x30\x34\x38\x12#\n\x15\x61ttention_temperature\x18\x03 \x01(\x02:\x04\x30.01\x12$\n\x16\x63ontext_feature_length\x18\x04 \x01(\x05:\x04\x32\x30\x35\x37\"\xc3\x02\n\x1a\x46\x61sterRcnnFeatureExtractor\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\'\n\x1b\x66irst_stage_features_stride\x18\x02 \x01(\x05:\x02\x31\x36\x12#\n\x14\x62\x61tch_norm_trainable\x18\x03 \x01(\x08:\x05\x66\x61lse\x12\x38\n\x10\x63onv_hyperparams\x18\x04 \x01(\x0b\x32\x1e.fasterrcnn.protos.Hyperparams\x12:\n+override_base_feature_extractor_hyperparams\x18\x05 \x01(\x08:\x05\x66\x61lse\x12\x1b\n\x0fpad_to_multiple\x18\x06 \x01(\x05:\x02\x33\x32\x12\x36\n\x03\x66pn\x18\x07 \x01(\x0b\x32).fasterrcnn.protos.FeaturePyramidNetworks')
  ,
  dependencies=[fasterrcnn_dot_protos_dot_anchor__generator__pb2.DESCRIPTOR,fasterrcnn_dot_protos_dot_box__predictor__pb2.DESCRIPTOR,fasterrcnn_dot_protos_dot_hyperparams__pb2.DESCRIPTOR,fasterrcnn_dot_protos_dot_image__resizer__pb2.DESCRIPTOR,fasterrcnn_dot_protos_dot_losses__pb2.DESCRIPTOR,fasterrcnn_dot_protos_dot_post__processing__pb2.DESCRIPTOR,fasterrcnn_dot_protos_dot_fpn__pb2.DESCRIPTOR,])




_FASTERRCNN = _descriptor.Descriptor(
  name='FasterRcnn',
  full_name='fasterrcnn.protos.FasterRcnn',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='number_of_stages', full_name='fasterrcnn.protos.FasterRcnn.number_of_stages', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='num_classes', full_name='fasterrcnn.protos.FasterRcnn.num_classes', index=1,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='image_resizer', full_name='fasterrcnn.protos.FasterRcnn.image_resizer', index=2,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='feature_extractor', full_name='fasterrcnn.protos.FasterRcnn.feature_extractor', index=3,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_anchor_generator', full_name='fasterrcnn.protos.FasterRcnn.first_stage_anchor_generator', index=4,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_atrous_rate', full_name='fasterrcnn.protos.FasterRcnn.first_stage_atrous_rate', index=5,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_box_predictor_conv_hyperparams', full_name='fasterrcnn.protos.FasterRcnn.first_stage_box_predictor_conv_hyperparams', index=6,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_box_predictor_kernel_size', full_name='fasterrcnn.protos.FasterRcnn.first_stage_box_predictor_kernel_size', index=7,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=3,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_box_predictor_depth', full_name='fasterrcnn.protos.FasterRcnn.first_stage_box_predictor_depth', index=8,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=512,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_minibatch_size', full_name='fasterrcnn.protos.FasterRcnn.first_stage_minibatch_size', index=9,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=256,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_positive_balance_fraction', full_name='fasterrcnn.protos.FasterRcnn.first_stage_positive_balance_fraction', index=10,
      number=12, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_nms_score_threshold', full_name='fasterrcnn.protos.FasterRcnn.first_stage_nms_score_threshold', index=11,
      number=13, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_nms_iou_threshold', full_name='fasterrcnn.protos.FasterRcnn.first_stage_nms_iou_threshold', index=12,
      number=14, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.7),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_max_proposals', full_name='fasterrcnn.protos.FasterRcnn.first_stage_max_proposals', index=13,
      number=15, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=300,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_localization_loss_weight', full_name='fasterrcnn.protos.FasterRcnn.first_stage_localization_loss_weight', index=14,
      number=16, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_objectness_loss_weight', full_name='fasterrcnn.protos.FasterRcnn.first_stage_objectness_loss_weight', index=15,
      number=17, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='initial_crop_size', full_name='fasterrcnn.protos.FasterRcnn.initial_crop_size', index=16,
      number=18, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='maxpool_kernel_size', full_name='fasterrcnn.protos.FasterRcnn.maxpool_kernel_size', index=17,
      number=19, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='maxpool_stride', full_name='fasterrcnn.protos.FasterRcnn.maxpool_stride', index=18,
      number=20, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_box_predictor', full_name='fasterrcnn.protos.FasterRcnn.second_stage_box_predictor', index=19,
      number=21, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_batch_size', full_name='fasterrcnn.protos.FasterRcnn.second_stage_batch_size', index=20,
      number=22, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=64,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_balance_fraction', full_name='fasterrcnn.protos.FasterRcnn.second_stage_balance_fraction', index=21,
      number=23, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.25),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_post_processing', full_name='fasterrcnn.protos.FasterRcnn.second_stage_post_processing', index=22,
      number=24, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_localization_loss_weight', full_name='fasterrcnn.protos.FasterRcnn.second_stage_localization_loss_weight', index=23,
      number=25, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_classification_loss_weight', full_name='fasterrcnn.protos.FasterRcnn.second_stage_classification_loss_weight', index=24,
      number=26, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_mask_prediction_loss_weight', full_name='fasterrcnn.protos.FasterRcnn.second_stage_mask_prediction_loss_weight', index=25,
      number=27, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='hard_example_miner', full_name='fasterrcnn.protos.FasterRcnn.hard_example_miner', index=26,
      number=28, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='second_stage_classification_loss', full_name='fasterrcnn.protos.FasterRcnn.second_stage_classification_loss', index=27,
      number=29, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='inplace_batchnorm_update', full_name='fasterrcnn.protos.FasterRcnn.inplace_batchnorm_update', index=28,
      number=30, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_matmul_crop_and_resize', full_name='fasterrcnn.protos.FasterRcnn.use_matmul_crop_and_resize', index=29,
      number=31, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='clip_anchors_to_image', full_name='fasterrcnn.protos.FasterRcnn.clip_anchors_to_image', index=30,
      number=32, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_matmul_gather_in_matcher', full_name='fasterrcnn.protos.FasterRcnn.use_matmul_gather_in_matcher', index=31,
      number=33, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_static_balanced_label_sampler', full_name='fasterrcnn.protos.FasterRcnn.use_static_balanced_label_sampler', index=32,
      number=34, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_static_shapes', full_name='fasterrcnn.protos.FasterRcnn.use_static_shapes', index=33,
      number=35, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='resize_masks', full_name='fasterrcnn.protos.FasterRcnn.resize_masks', index=34,
      number=36, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_static_shapes_for_eval', full_name='fasterrcnn.protos.FasterRcnn.use_static_shapes_for_eval', index=35,
      number=37, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_partitioned_nms_in_first_stage', full_name='fasterrcnn.protos.FasterRcnn.use_partitioned_nms_in_first_stage', index=36,
      number=38, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='return_raw_detections_during_predict', full_name='fasterrcnn.protos.FasterRcnn.return_raw_detections_during_predict', index=37,
      number=39, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='use_combined_nms_in_first_stage', full_name='fasterrcnn.protos.FasterRcnn.use_combined_nms_in_first_stage', index=38,
      number=40, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='output_final_box_features', full_name='fasterrcnn.protos.FasterRcnn.output_final_box_features', index=39,
      number=42, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='context_config', full_name='fasterrcnn.protos.FasterRcnn.context_config', index=40,
      number=41, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=318,
  serialized_end=2289,
)


_CONTEXT = _descriptor.Descriptor(
  name='Context',
  full_name='fasterrcnn.protos.Context',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='max_num_context_features', full_name='fasterrcnn.protos.Context.max_num_context_features', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=2000,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='attention_bottleneck_dimension', full_name='fasterrcnn.protos.Context.attention_bottleneck_dimension', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=2048,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='attention_temperature', full_name='fasterrcnn.protos.Context.attention_temperature', index=2,
      number=3, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(0.01),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='context_feature_length', full_name='fasterrcnn.protos.Context.context_feature_length', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=2057,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2292,
  serialized_end=2462,
)


_FASTERRCNNFEATUREEXTRACTOR = _descriptor.Descriptor(
  name='FasterRcnnFeatureExtractor',
  full_name='fasterrcnn.protos.FasterRcnnFeatureExtractor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='fasterrcnn.protos.FasterRcnnFeatureExtractor.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='first_stage_features_stride', full_name='fasterrcnn.protos.FasterRcnnFeatureExtractor.first_stage_features_stride', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=16,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='batch_norm_trainable', full_name='fasterrcnn.protos.FasterRcnnFeatureExtractor.batch_norm_trainable', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='conv_hyperparams', full_name='fasterrcnn.protos.FasterRcnnFeatureExtractor.conv_hyperparams', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='override_base_feature_extractor_hyperparams', full_name='fasterrcnn.protos.FasterRcnnFeatureExtractor.override_base_feature_extractor_hyperparams', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='pad_to_multiple', full_name='fasterrcnn.protos.FasterRcnnFeatureExtractor.pad_to_multiple', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=32,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='fpn', full_name='fasterrcnn.protos.FasterRcnnFeatureExtractor.fpn', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=2465,
  serialized_end=2788,
)

_FASTERRCNN.fields_by_name['image_resizer'].message_type = fasterrcnn_dot_protos_dot_image__resizer__pb2._IMAGERESIZER
_FASTERRCNN.fields_by_name['feature_extractor'].message_type = _FASTERRCNNFEATUREEXTRACTOR
_FASTERRCNN.fields_by_name['first_stage_anchor_generator'].message_type = fasterrcnn_dot_protos_dot_anchor__generator__pb2._ANCHORGENERATOR
_FASTERRCNN.fields_by_name['first_stage_box_predictor_conv_hyperparams'].message_type = fasterrcnn_dot_protos_dot_hyperparams__pb2._HYPERPARAMS
_FASTERRCNN.fields_by_name['second_stage_box_predictor'].message_type = fasterrcnn_dot_protos_dot_box__predictor__pb2._BOXPREDICTOR
_FASTERRCNN.fields_by_name['second_stage_post_processing'].message_type = fasterrcnn_dot_protos_dot_post__processing__pb2._POSTPROCESSING
_FASTERRCNN.fields_by_name['hard_example_miner'].message_type = fasterrcnn_dot_protos_dot_losses__pb2._HARDEXAMPLEMINER
_FASTERRCNN.fields_by_name['second_stage_classification_loss'].message_type = fasterrcnn_dot_protos_dot_losses__pb2._CLASSIFICATIONLOSS
_FASTERRCNN.fields_by_name['context_config'].message_type = _CONTEXT
_FASTERRCNNFEATUREEXTRACTOR.fields_by_name['conv_hyperparams'].message_type = fasterrcnn_dot_protos_dot_hyperparams__pb2._HYPERPARAMS
_FASTERRCNNFEATUREEXTRACTOR.fields_by_name['fpn'].message_type = fasterrcnn_dot_protos_dot_fpn__pb2._FEATUREPYRAMIDNETWORKS
DESCRIPTOR.message_types_by_name['FasterRcnn'] = _FASTERRCNN
DESCRIPTOR.message_types_by_name['Context'] = _CONTEXT
DESCRIPTOR.message_types_by_name['FasterRcnnFeatureExtractor'] = _FASTERRCNNFEATUREEXTRACTOR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

FasterRcnn = _reflection.GeneratedProtocolMessageType('FasterRcnn', (_message.Message,), dict(
  DESCRIPTOR = _FASTERRCNN,
  __module__ = 'fasterrcnn.protos.faster_rcnn_pb2'
  # @@protoc_insertion_point(class_scope:fasterrcnn.protos.FasterRcnn)
  ))
_sym_db.RegisterMessage(FasterRcnn)

Context = _reflection.GeneratedProtocolMessageType('Context', (_message.Message,), dict(
  DESCRIPTOR = _CONTEXT,
  __module__ = 'fasterrcnn.protos.faster_rcnn_pb2'
  # @@protoc_insertion_point(class_scope:fasterrcnn.protos.Context)
  ))
_sym_db.RegisterMessage(Context)

FasterRcnnFeatureExtractor = _reflection.GeneratedProtocolMessageType('FasterRcnnFeatureExtractor', (_message.Message,), dict(
  DESCRIPTOR = _FASTERRCNNFEATUREEXTRACTOR,
  __module__ = 'fasterrcnn.protos.faster_rcnn_pb2'
  # @@protoc_insertion_point(class_scope:fasterrcnn.protos.FasterRcnnFeatureExtractor)
  ))
_sym_db.RegisterMessage(FasterRcnnFeatureExtractor)


# @@protoc_insertion_point(module_scope)
