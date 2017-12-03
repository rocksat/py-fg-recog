from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from lib.datasets.format_cub_dataset_parts import create_image_sizes_file, format_dataset, create_validation_split
from lib.tfrecords.create_tfrecords import create

def main(cub_dataset_dir, cub_image_dir, output_dir):
    # we need to create a file containing the size of each image in the dataset. 
    # you only need to do this once. scipy is required for this method. 
    # Alternatively, you can create this file yourself. 
    # Each line of the file should have <image_id> <width> <height>
    create_image_sizes_file(cub_dataset_dir, cub_image_dir)

    # Now we can create the datasets
    train, test = format_dataset(cub_dataset_dir, cub_image_dir)
    train, val = create_validation_split(train, fraction_per_class=0.1, shuffle=True)


    # Create tfrecord
    train_errors = create(dataset=train, dataset_name="train", output_directory=output_dir,
                          num_shards=5, num_threads=5, shuffle=True)

    val_errors = create(dataset=val, dataset_name="val", output_directory=output_dir,
                        num_shards=1, num_threads=1, shuffle=True)

    test_errors = create(dataset=test, dataset_name="test", output_directory=output_dir,
                        num_shards=5, num_threads=5, shuffle=True)

def parse_args():
    
    parser = argparse.ArgumentParser(description='Create TFRecord files for CUB-200-2011 dataset')

    parser.add_argument('--cub_dataset_dir', dest='cub_dataset_dir',
                        help='Path to the dataset folder', type=str,
                        default='./data/cub')

    parser.add_argument('--cub_image_dir', dest='cub_image_dir',
                        help='Directory for CUB-200-2011 images', type=str,
                        default='./data/cub/images')

    parser.add_argument('--output_dir', dest='output_dir',
                        help='Directory for the tfrecords', type=str,
                        default='./data/tfrecords/cub')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args.cub_dataset_dir, args.cub_image_dir, args.output_dir)
    
