export DATASET_DIR=./data/tfrecords/aircraft
export EXPERIMENT_DIR=./experiments
export IMAGENET_PRETRAINED_MODEL=./data/imagenet_models/inception_v3.ckpt

# Train all of the weigths, using the finetuned model as a starting point
CUDA_VISIBLE_DEVICES=0 python lib/classification/train.py \
--tfrecords $DATASET_DIR/train* \
--logdir $EXPERIMENT_DIR/logs/aircraft/ \
--config $EXPERIMENT_DIR/cfgs/aircraft_image_config_train.yaml \
--pretrained_model $EXPERIMENT_DIR/logs/aircraft/finetune \
