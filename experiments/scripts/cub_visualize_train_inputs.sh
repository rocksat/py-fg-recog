export DATASET_DIR=./data/tfrecords/cub
export EXPERIMENT_DIR=./experiments/cfgs
export LOG_DIR=./experiments/logs/cub
export IMAGENET_PRETRAINED_MODEL=./data/imagenet_models/inception_v3.ckpt

# Visualize the inputs to the network
CUDA_VISIBLE_DEVICES=1 python lib/preprocessing/visualize_train_inputs.py \
--tfrecords $DATASET_DIR/train* \
--config $EXPERIMENT_DIR/cub_image_config_train.yaml \
--text_labels
