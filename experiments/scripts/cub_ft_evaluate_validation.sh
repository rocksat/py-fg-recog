export DATASET_DIR=./data/tfrecords/cub
export EXPERIMENT_DIR=./experiments
export IMAGENET_PRETRAINED_MODEL=./data/imagenet_models/inception_v3.ckpt

# Evaluate the finetuned model with the validation data
CUDA_VISIBLE_DEVICES='1' python lib/classification/test.py \
--tfrecords $DATASET_DIR/val* \
--save_dir $EXPERIMENT_DIR/logs/cub/finetune/val_summaries \
--checkpoint_path $EXPERIMENT_DIR/logs/cub/finetune \
--config $EXPERIMENT_DIR/cfgs/cub_image_config_test.yaml \
--batch_size 20 \
--batches 30 \
--eval_interval_secs 180
