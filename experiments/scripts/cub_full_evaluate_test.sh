export DATASET_DIR=./data/tfrecords/cub
export EXPERIMENT_DIR=./experiments
export IMAGENET_PRETRAINED_MODEL=./data/imagenet_models/inception_v3.ckpt

# Evaluate the model using the test data
CUDA_VISIBLE_DEVICES='0' python lib/classification/test.py \
--tfrecords $DATASET_DIR/test* \
--save_dir $EXPERIMENT_DIR/logs/cub/test_summaries \
--checkpoint_path $EXPERIMENT_DIR/logs/cub \
--config $EXPERIMENT_DIR/cfgs/cub_image_config_test.yaml \
--batch_size 1 \
--batches 5794
