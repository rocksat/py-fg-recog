from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from lib.datasets.format_aircraft_dataset import create_image_sizes_file, format_dataset, create_validation_split
from lib.tfrecords.create_tfrecords import create

def main(aircraft_dataset_dir, aircraft_image_dir, output_dir):
    # we need to create a file containing the size of each image in the dataset.
    # you only need to do this once. scipy is required for this method.
    # Alternatively, you can create this file yourself.
    # Each line of the file should have <image_id> <width> <height>
    # create_image_sizes_file(aircraft_dataset_dir, aircraft_image_dir)
    # Now we can create the datasets
    train, test = format_dataset(aircraft_dataset_dir, aircraft_image_dir)
    train, val = create_validation_split(train, fraction_per_class=0.1, shuffle=True)

    # Create tfrecord
    train_errors = create(dataset=train, dataset_name="train", output_directory=output_dir,
                          num_shards=6, num_threads=2, shuffle=True)

    val_errors = create(dataset=val, dataset_name="val", output_directory=output_dir,
                        num_shards=1, num_threads=1, shuffle=True)

    test_errors = create(dataset=test, dataset_name="test", output_directory=output_dir,
                        num_shards=4, num_threads=2, shuffle=True)


def parse_args():

    parser = argparse.ArgumentParser(description='Create TFRecord files for FGVC-aircraft dataset')

    parser.add_argument('--aircraft_dataset_dir', dest='aircraft_dataset_dir',
                        help='Path to the dataset folder', type=str,
                        default='./data/aircraft/data')

    parser.add_argument('--aircraft_image_dir', dest='aircraft_image_dir',
                        help='Directory for FGVC-aircraft images', type=str,
                        default='./data/aircraft/data/images')

    parser.add_argument('--output_dir', dest='output_dir',
                        help='Directory for the tfrecords', type=str,
                        default='./data/tfrecords/aircraft')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args.aircraft_dataset_dir, args.aircraft_image_dir, args.output_dir)
