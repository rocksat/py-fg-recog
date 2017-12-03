export DATASET_DIR=./data/tfrecords/cub
export EXPERIMENT_DIR=./experiments
export IMAGENET_PRETRAINED_MODEL=./data/imagenet_models/inception_v3.ckpt

# Warm up training phase
CUDA_VISIBLE_DEVICES='0' python lib/classification/train.py \
--tfrecords $DATASET_DIR/train* \
--logdir $EXPERIMENT_DIR/logs/cub/finetune \
--config $EXPERIMENT_DIR/cfgs/cub_image_config_train.yaml \
--pretrained_model $IMAGENET_PRETRAINED_MODEL \
--trainable_scopes InceptionV3/Logits InceptionV3/AuxLogits \
--checkpoint_exclude_scopes InceptionV3/Logits InceptionV3/AuxLogits \
--learning_rate_decay_type fixed \
--lr 0.01
