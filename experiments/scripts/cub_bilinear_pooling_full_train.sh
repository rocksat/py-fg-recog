export DATASET_DIR=./data/tfrecords/cub
export EXPERIMENT_DIR=./experiments
export IMAGENET_PRETRAINED_MODEL=./data/imagenet_models/inception_v3.ckpt

# Train all of the weigths, using the finetuned model as a starting point
CUDA_VISIBLE_DEVICES='0' python lib/compact_bilinear_pooling/train.py \
--tfrecords $DATASET_DIR/train* \
--logdir $EXPERIMENT_DIR/logs/cub/ \
--config $EXPERIMENT_DIR/cfgs/cub_image_config_train.yaml \
--pretrained_model $EXPERIMENT_DIR/logs/cub/finetune \
