import tensorflow as tf
from functools import partial

from fasterrcnn.builders import hyperparams_builder
from fasterrcnn.builders import box_predictor_builder
from fasterrcnn.builders import image_resizer_builder
from fasterrcnn.models import faster_rcnn_resnet_keras_feature_extractor as frcnn_resnet_keras
# from builders import anchor_generator_builder
from fasterrcnn.anchor_generators import grid_anchor_generator
from fasterrcnn.core import target_assigner
from fasterrcnn.core import balanced_positive_negative_sampler as sampler
from fasterrcnn.core import post_processing
from fasterrcnn.builders import post_processing_builder
from fasterrcnn.builders import losses_builder
from fasterrcnn.utils import spatial_transform_ops as spatial_ops


import faster_rcnn_meta_arch


def anchor_generator_builder(grid_anchor_generator_config):
    """Builds an anchor generator based on the config.

    Args:
    anchor_generator_config: An anchor_generator.proto object containing the
      config for the desired anchor generator.

    Returns:
    Anchor generator based on the config.

    Raises:
    ValueError: On empty anchor generator proto.
    """

    # grid_anchor_generator_config = anchor_generator_config.grid_anchor_generator
    return grid_anchor_generator.GridAnchorGenerator(
        scales=[float(scale) for scale in grid_anchor_generator_config.scales],
        aspect_ratios=[float(aspect_ratio)
                       for aspect_ratio
                       in grid_anchor_generator_config.aspect_ratios],
        base_anchor_size=[grid_anchor_generator_config.height,
                          grid_anchor_generator_config.width],
        anchor_stride=[grid_anchor_generator_config.height_stride,
                       grid_anchor_generator_config.width_stride],
        anchor_offset=[grid_anchor_generator_config.height_offset,
                       grid_anchor_generator_config.width_offset])


def _build_faster_rcnn_keras_feature_extractor(
    feature_extractor_config,
    inplace_batchnorm_update=False):
  """Builds a faster_rcnn_meta_arch.FasterRCNNKerasFeatureExtractor from config.

  Args:
    feature_extractor_config: A FasterRcnnFeatureExtractor proto config from
      faster_rcnn.proto.
    is_training: True if this feature extractor is being built for training.
    inplace_batchnorm_update: Whether to update batch_norm inplace during
      training. This is required for batch norm to work correctly on TPUs. When
      this is false, user must add a control dependency on
      tf.GraphKeys.UPDATE_OPS for train/loss op in order to update the batch
      norm moving average parameters.

  Returns:
    faster_rcnn_meta_arch.FasterRCNNKerasFeatureExtractor based on config.

  Raises:
    ValueError: On invalid feature extractor type.
  """
  if inplace_batchnorm_update:
    raise ValueError('inplace batchnorm updates not supported.')
  # feature_type = feature_extractor_config.type
  first_stage_features_stride = (
      feature_extractor_config.first_stage_features_stride)
  # batch_norm_trainable = feature_extractor_config.batch_norm_trainable

  # if feature_type not in FASTER_RCNN_KERAS_FEATURE_EXTRACTOR_CLASS_MAP:
  #   raise ValueError('Unknown Faster R-CNN feature_extractor: {}'.format(
  #       feature_type))
  # feature_extractor_class = FASTER_RCNN_KERAS_FEATURE_EXTRACTOR_CLASS_MAP[
  #     feature_type]

  feature_extractor_class = frcnn_resnet_keras.FasterRCNNResnet50KerasFeatureExtractor

  kwargs = {}

  if feature_extractor_config.HasField('conv_hyperparams'):
    kwargs.update({
        'conv_hyperparams':
            hyperparams_builder.KerasLayerHyperparams(
                feature_extractor_config.conv_hyperparams),
        'override_base_feature_extractor_hyperparams':
            feature_extractor_config.override_base_feature_extractor_hyperparams
    })

  if feature_extractor_config.HasField('fpn'):
    kwargs.update({
        'fpn_min_level':
            feature_extractor_config.fpn.min_level,
        'fpn_max_level':
            feature_extractor_config.fpn.max_level,
        'additional_layer_depth':
            feature_extractor_config.fpn.additional_layer_depth,
    })

  return feature_extractor_class(first_stage_features_stride, **kwargs)



def _build_faster_rcnn_model(frcnn_config, is_training, add_summaries, num_classes, min_dim, max_dim):
  """Builds a Faster R-CNN or R-FCN detection model based on the model config.

  Builds R-FCN model if the second_stage_box_predictor in the config is of type
  `rfcn_box_predictor` else builds a Faster R-CNN model.

  Args:
    frcnn_config: A faster_rcnn.proto object containing the config for the
      desired FasterRCNNMetaArch or RFCNMetaArch.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tf summaries in the model.

  Returns:
    FasterRCNNMetaArch based on the config.

  Raises:
    ValueError: If frcnn_config.type is not recognized (i.e. not registered in
      model_class_map).
  """

  # Do later same as retinanet
  image_resizer_fn = image_resizer_builder.build(min_dimension=min_dim, max_dimension=max_dim)

  # if is_keras:
  feature_extractor = _build_faster_rcnn_keras_feature_extractor(
        frcnn_config.feature_extractor,
        inplace_batchnorm_update=frcnn_config.inplace_batchnorm_update)
  # else:
  #   feature_extractor = _build_faster_rcnn_feature_extractor(
  #       frcnn_config.feature_extractor, is_training,
  #       inplace_batchnorm_update=frcnn_config.inplace_batchnorm_update)

  number_of_stages = frcnn_config.number_of_stages
  first_stage_anchor_generator = anchor_generator_builder(
      frcnn_config.first_stage_anchor_generator.grid_anchor_generator)

  first_stage_target_assigner = target_assigner.create_target_assigner(
      'FasterRCNN',
      'proposal',
      use_matmul_gather=True)

  first_stage_atrous_rate = frcnn_config.first_stage_atrous_rate

  # if is_keras:
  first_stage_box_predictor_arg_scope_fn = (
        hyperparams_builder.KerasLayerHyperparams(
            frcnn_config.first_stage_box_predictor_conv_hyperparams))
  # else:
  #   first_stage_box_predictor_arg_scope_fn = hyperparams_builder.build(
  #       frcnn_config.first_stage_box_predictor_conv_hyperparams, is_training)

  first_stage_box_predictor_kernel_size = (
      frcnn_config.first_stage_box_predictor_kernel_size)
  first_stage_box_predictor_depth = frcnn_config.first_stage_box_predictor_depth
  first_stage_minibatch_size = frcnn_config.first_stage_minibatch_size
  use_static_shapes = frcnn_config.use_static_shapes and (
      frcnn_config.use_static_shapes_for_eval or is_training)
  first_stage_sampler = sampler.BalancedPositiveNegativeSampler(
      positive_fraction=frcnn_config.first_stage_positive_balance_fraction,
      is_static=(frcnn_config.use_static_balanced_label_sampler and
                 use_static_shapes))
  first_stage_max_proposals = frcnn_config.first_stage_max_proposals
  if (frcnn_config.first_stage_nms_iou_threshold < 0 or
      frcnn_config.first_stage_nms_iou_threshold > 1.0):
    raise ValueError('iou_threshold not in [0, 1.0].')
  if (is_training and frcnn_config.second_stage_batch_size >
      first_stage_max_proposals):
    raise ValueError('second_stage_batch_size should be no greater than '
                     'first_stage_max_proposals.')
  first_stage_non_max_suppression_fn = partial(
      post_processing.batch_multiclass_non_max_suppression,
      score_thresh=frcnn_config.first_stage_nms_score_threshold,
      iou_thresh=frcnn_config.first_stage_nms_iou_threshold,
      max_size_per_class=frcnn_config.first_stage_max_proposals,
      max_total_size=frcnn_config.first_stage_max_proposals,
      use_static_shapes=use_static_shapes,
      use_partitioned_nms=frcnn_config.use_partitioned_nms_in_first_stage,
      use_combined_nms=frcnn_config.use_combined_nms_in_first_stage)
  first_stage_loc_loss_weight = (
      frcnn_config.first_stage_localization_loss_weight)
  first_stage_obj_loss_weight = frcnn_config.first_stage_objectness_loss_weight

  initial_crop_size = frcnn_config.initial_crop_size
  maxpool_kernel_size = frcnn_config.maxpool_kernel_size
  maxpool_stride = frcnn_config.maxpool_stride

  second_stage_target_assigner = target_assigner.create_target_assigner(
      'FasterRCNN',
      'detection',
      use_matmul_gather=frcnn_config.use_matmul_gather_in_matcher)

  # if is_keras:
  second_stage_box_predictor = box_predictor_builder.build_keras(
        hyperparams_builder.KerasLayerHyperparams,
        freeze_batchnorm=False,
        inplace_batchnorm_update=False,
        num_predictions_per_location_list=[1],
        box_predictor_config=frcnn_config.second_stage_box_predictor,
        is_training=is_training,
        num_classes=num_classes)
  # else:
  #   second_stage_box_predictor = box_predictor_builder.build(
  #       hyperparams_builder.build,
  #       frcnn_config.second_stage_box_predictor,
  #       is_training=is_training,
  #       num_classes=num_classes)

  second_stage_batch_size = frcnn_config.second_stage_batch_size
  second_stage_sampler = sampler.BalancedPositiveNegativeSampler(
      positive_fraction=frcnn_config.second_stage_balance_fraction,
      is_static=(frcnn_config.use_static_balanced_label_sampler and
                 use_static_shapes))
  (second_stage_non_max_suppression_fn, second_stage_score_conversion_fn
  ) = post_processing_builder.build(frcnn_config.second_stage_post_processing)
  second_stage_localization_loss_weight = (
      frcnn_config.second_stage_localization_loss_weight)
  second_stage_classification_loss = (
      losses_builder.build_faster_rcnn_classification_loss(
          frcnn_config.second_stage_classification_loss))
  second_stage_classification_loss_weight = (
      frcnn_config.second_stage_classification_loss_weight)
  second_stage_mask_prediction_loss_weight = (
      frcnn_config.second_stage_mask_prediction_loss_weight)

  hard_example_miner = None
  if frcnn_config.HasField('hard_example_miner'):
    hard_example_miner = losses_builder.build_hard_example_miner(
        frcnn_config.hard_example_miner,
        second_stage_classification_loss_weight,
        second_stage_localization_loss_weight)

  crop_and_resize_fn = (
      spatial_ops.multilevel_matmul_crop_and_resize
      if frcnn_config.use_matmul_crop_and_resize
      else spatial_ops.multilevel_native_crop_and_resize)
  clip_anchors_to_image = (
      frcnn_config.clip_anchors_to_image)

  common_kwargs = {
      'is_training':
          is_training,
      'num_classes':
          num_classes,
      'image_resizer_fn':
          image_resizer_fn,
      'feature_extractor':
          feature_extractor,
      'number_of_stages':
          number_of_stages,
      'first_stage_anchor_generator':
          first_stage_anchor_generator,
      'first_stage_target_assigner':
          first_stage_target_assigner,
      'first_stage_atrous_rate':
          first_stage_atrous_rate,
      'first_stage_box_predictor_arg_scope_fn':
          first_stage_box_predictor_arg_scope_fn,
      'first_stage_box_predictor_kernel_size':
          first_stage_box_predictor_kernel_size,
      'first_stage_box_predictor_depth':
          first_stage_box_predictor_depth,
      'first_stage_minibatch_size':
          first_stage_minibatch_size,
      'first_stage_sampler':
          first_stage_sampler,
      'first_stage_non_max_suppression_fn':
          first_stage_non_max_suppression_fn,
      'first_stage_max_proposals':
          first_stage_max_proposals,
      'first_stage_localization_loss_weight':
          first_stage_loc_loss_weight,
      'first_stage_objectness_loss_weight':
          first_stage_obj_loss_weight,
      'second_stage_target_assigner':
          second_stage_target_assigner,
      'second_stage_batch_size':
          second_stage_batch_size,
      'second_stage_sampler':
          second_stage_sampler,
      'second_stage_non_max_suppression_fn':
          second_stage_non_max_suppression_fn,
      'second_stage_score_conversion_fn':
          second_stage_score_conversion_fn,
      'second_stage_localization_loss_weight':
          second_stage_localization_loss_weight,
      'second_stage_classification_loss':
          second_stage_classification_loss,
      'second_stage_classification_loss_weight':
          second_stage_classification_loss_weight,
      'hard_example_miner':
          hard_example_miner,
      'add_summaries':
          add_summaries,
      'crop_and_resize_fn':
          crop_and_resize_fn,
      'clip_anchors_to_image':
          clip_anchors_to_image,
      'use_static_shapes':
          use_static_shapes,
      'resize_masks':
          frcnn_config.resize_masks,
      'return_raw_detections_during_predict':
          frcnn_config.return_raw_detections_during_predict,
      'output_final_box_features':
          frcnn_config.output_final_box_features
  }

  # if isinstance(second_stage_box_predictor, rfcn_keras_box_predictor.RfcnKerasBoxPredictor):
  #   return rfcn_meta_arch.RFCNMetaArch(
  #       second_stage_rfcn_box_predictor=second_stage_box_predictor,
  #       **common_kwargs)
  # elif frcnn_config.HasField('context_config'):
  #   context_config = frcnn_config.context_config
  #   common_kwargs.update({
  #       'attention_bottleneck_dimension':
  #           context_config.attention_bottleneck_dimension,
  #       'attention_temperature':
  #           context_config.attention_temperature
  #   })
  #   return context_rcnn_meta_arch.ContextRCNNMetaArch(
  #       initial_crop_size=initial_crop_size,
  #       maxpool_kernel_size=maxpool_kernel_size,
  #       maxpool_stride=maxpool_stride,
  #       second_stage_mask_rcnn_box_predictor=second_stage_box_predictor,
  #       second_stage_mask_prediction_loss_weight=(
  #           second_stage_mask_prediction_loss_weight),
  #       **common_kwargs)
  # else:
  return faster_rcnn_meta_arch.FasterRCNNMetaArch(
        initial_crop_size=initial_crop_size,
        maxpool_kernel_size=maxpool_kernel_size,
        maxpool_stride=maxpool_stride,
        second_stage_mask_rcnn_box_predictor=second_stage_box_predictor,
        second_stage_mask_prediction_loss_weight=(
            second_stage_mask_prediction_loss_weight),
        **common_kwargs)


def build(model_config, is_training, num_classes, min_dim, max_dim, add_summaries=True):
    """Builds a DetectionModel based on the model config.

    Args:
    model_config: A model.proto object containing the config for the desired
      DetectionModel.
    is_training: True if this model is being built for training purposes.
    add_summaries: Whether to add tensorflow summaries in the model graph.
    Returns:
    DetectionModel based on the config.

    Raises:
    ValueError: On invalid meta architecture or model.
    """

    return _build_faster_rcnn_model(getattr(model_config, 'faster_rcnn'), is_training,
                      add_summaries, num_classes, min_dim, max_dim)


