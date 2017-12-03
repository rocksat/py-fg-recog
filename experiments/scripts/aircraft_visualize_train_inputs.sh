export DATASET_DIR=./data/tfrecords/aircraft
export EXPERIMENT_DIR=./experiments

# Visualize the inputs to the network
CUDA_VISIBLE_DEVICES=1 python lib/classification/visualize_train_inputs.py \
--tfrecords $DATASET_DIR/train* \
--config $EXPERIMENT_DIR/cfgs/aircraft_image_config_train.yaml \
--text_labels

