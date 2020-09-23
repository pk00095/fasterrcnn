import tensorflow as tf
from fasterrcnn.utils import config_util
from model_builder import build as model_build
import inputs
import pdb

def _compute_losses_and_predictions_dicts(
    model, features, labels,
    add_regularization_loss=True):
  """Computes the losses dict and predictions dict for a model on inputs.

  Args:
    model: a DetectionModel (based on Keras).
    features: Dictionary of feature tensors from the input dataset.
      Should be in the format output by `inputs.train_input` and
      `inputs.eval_input`.
        features[fields.InputDataFields.image] is a [batch_size, H, W, C]
          float32 tensor with preprocessed images.
        features[HASH_KEY] is a [batch_size] int32 tensor representing unique
          identifiers for the images.
        features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]
          int32 tensor representing the true image shapes, as preprocessed
          images could be padded.
        features[fields.InputDataFields.original_image] (optional) is a
          [batch_size, H, W, C] float32 tensor with original images.
    labels: A dictionary of groundtruth tensors post-unstacking. The original
      labels are of the form returned by `inputs.train_input` and
      `inputs.eval_input`. The shapes may have been modified by unstacking with
      `model_lib.unstack_batch`. However, the dictionary includes the following
      fields.
        labels[fields.InputDataFields.num_groundtruth_boxes] is a
          int32 tensor indicating the number of valid groundtruth boxes
          per image.
        labels[fields.InputDataFields.groundtruth_boxes] is a float32 tensor
          containing the corners of the groundtruth boxes.
        labels[fields.InputDataFields.groundtruth_classes] is a float32
          one-hot tensor of classes.
        labels[fields.InputDataFields.groundtruth_weights] is a float32 tensor
          containing groundtruth weights for the boxes.
        -- Optional --
        labels[fields.InputDataFields.groundtruth_instance_masks] is a
          float32 tensor containing only binary values, which represent
          instance masks for objects.
        labels[fields.InputDataFields.groundtruth_keypoints] is a
          float32 tensor containing keypoints for each box.
        labels[fields.InputDataFields.groundtruth_dp_num_points] is an int32
          tensor with the number of sampled DensePose points per object.
        labels[fields.InputDataFields.groundtruth_dp_part_ids] is an int32
          tensor with the DensePose part ids (0-indexed) per object.
        labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
          float32 tensor with the DensePose surface coordinates.
        labels[fields.InputDataFields.groundtruth_group_of] is a tf.bool tensor
          containing group_of annotations.
        labels[fields.InputDataFields.groundtruth_labeled_classes] is a float32
          k-hot tensor of classes.
        labels[fields.InputDataFields.groundtruth_track_ids] is a int32
          tensor of track IDs.
    add_regularization_loss: Whether or not to include the model's
      regularization loss in the losses dictionary.

  Returns:
    A tuple containing the losses dictionary (with the total loss under
    the key 'Loss/total_loss'), and the predictions dictionary produced by
    `model.predict`.

  """
  model_lib.provide_groundtruth(model, labels)
  preprocessed_images = features[fields.InputDataFields.image]

  prediction_dict = model.predict(
      preprocessed_images,
      features[fields.InputDataFields.true_image_shape],
      **model.get_side_inputs(features))
  prediction_dict = ops.bfloat16_to_float32_nested(prediction_dict)

  losses_dict = model.loss(
      prediction_dict, features[fields.InputDataFields.true_image_shape])
  losses = [loss_tensor for loss_tensor in losses_dict.values()]
  if add_regularization_loss:
    # TODO(kaftan): As we figure out mixed precision & bfloat 16, we may
    ## need to convert these regularization losses from bfloat16 to float32
    ## as well.
    regularization_losses = model.regularization_losses()
    if regularization_losses:
      regularization_losses = ops.bfloat16_to_float32_nested(
          regularization_losses)
      regularization_loss = tf.add_n(
          regularization_losses, name='regularization_loss')
      losses.append(regularization_loss)
      losses_dict['Loss/regularization_loss'] = regularization_loss

  total_loss = tf.add_n(losses, name='total_loss')
  losses_dict['Loss/total_loss'] = total_loss

  return losses_dict, prediction_dict

if __name__ == '__main__':

    from fasterrcnn.protos import input_reader_pb2


    pipeline_config_path='faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.config'
    configs = config_util.get_configs_from_pipeline_file(
      pipeline_config_path)
    train_config = configs['train_config']
    model_config = configs['model']
    train_input_config = configs['train_input_config']

    detection_model = model_build(model_config, True, num_classes=4, min_dim=640, max_dim=640)

    # print(train_config.add_regularization_loss)

    train_input = inputs.train_input(
          train_config=train_config,
          train_input_config=train_input_config,
          model_config=model_config,
          batch_size=2,
          num_classes=90,
          min_dim=640,
          max_dim=640)

    print(train_input)

    # detection_model._is_training = is_training  # pylint: disable=protected-access
    # tf.keras.backend.set_learning_phase(is_training)
    # # pdb.set_trace()

    # losses_dict, _ = _compute_losses_and_predictions_dicts(
    #     detection_model, features, labels, add_regularization_loss)