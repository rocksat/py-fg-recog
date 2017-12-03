import os
import random
import sys
from collections import Counter

def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'variants.txt')) as f:
        class_id = 0
        for line in f:
            class_id += 1
            names[class_id] = line
    return names

def load_bounding_box_annotations(dataset_path=''):
    bboxes = {}

    with open(os.path.join(dataset_path, 'images_box.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            bbox = map(int, map(float, pieces[1:]))
            bboxes[image_id] = bbox

    return bboxes

def load_image_paths(dataset_path='', path_prefix=''):
    paths = {}

    with open(os.path.join(dataset_path, 'images_box.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            path = os.path.join(path_prefix, image_id + '.jpg')
            paths[image_id] = path

    return paths


def load_image_labels(dataset_path='', filename=''):
    labels = {}

    with open(os.path.join(dataset_path, filename)) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            class_id = ' '.join(pieces[1:])
            labels[image_id] = class_id

    return labels

# Not the best python code etiquette, but trying to keep everything self contained...
def create_image_sizes_file(dataset_path, image_path_prefix):

    from scipy.misc import imread

    image_paths = load_image_paths(dataset_path, image_path_prefix)
    image_sizes = []
    for image_id, image_path in image_paths.iteritems():
        im = imread(image_path)
        image_sizes.append([image_id, im.shape[1], im.shape[0]])

    with open(os.path.join(dataset_path, 'sizes.txt'), 'w') as f:
        for image_id, w, h in image_sizes:
            f.write("%s %d %d\n" % (str(image_id), w, h))

def load_image_sizes(dataset_path=''):
    sizes = {}

    with open(os.path.join(dataset_path, 'sizes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            image_id = pieces[0]
            width, height = map(int, pieces[1:])
            sizes[image_id] = [width, height]
    return sizes

def format_labels(image_labels):
    '''
    Convert the image labels to be integers between [0, num classes]

    Returns:
        condensed_image_labels = {image_id: new_label}
        new_id_to_original_id_map = {new_label: original_label}
    '''

    label_values = list(set(image_labels.values()))
    label_values.sort()
    condensed_image_labels = dict([(image_id, label_values.index(label))
                                    for image_id, label in
                                    image_labels.iteritems()])

    new_id_to_original_id_map = dict([[label_values.index(label), label] for label in label_values])

    return condensed_image_labels, new_id_to_original_id_map


def format_dataset(dataset_path, image_path_prefix):
    '''
    Load in a dataset (that have been saved in the CUB format) and store in a format to be written to the tfrecord file
    '''

    # class_names = load_class_names(dataset_path)
    image_paths = load_image_paths(dataset_path, image_path_prefix)
    image_sizes = load_image_sizes(dataset_path)
    image_bboxes = load_bounding_box_annotations(dataset_path)

    # load image labels
    train_labels, new_label_to_original_label_map = format_labels(load_image_labels(dataset_path, 'images_variant_trainval.txt'))
    test_labels, _ = format_labels(load_image_labels(dataset_path, 'images_variant_test.txt'))

    train_data = []
    test_data = []

    for image_labels, data_store in [(train_labels, train_data), (test_labels, test_data)]:
        for image_id in image_labels:

            width, height = image_sizes[image_id]
            width = float(width)
            height = float(height)

            x, y, w, h = image_bboxes[image_id]
            x1 = max(x / width,        0.)
            x2 = min((x + w) / width,  1.)
            y1 = max(y / height,       0.)
            y2 = min((y + h) / height, 1.)

            # we have no parts annotation
            parts_x = []
            parts_y = []
            parts_v = []

            data_store.append({
                "filename" : image_paths[image_id],
                "id" : image_id,
                "class" : {
                    "label" : image_labels[image_id],
                    "text": new_label_to_original_label_map[image_labels[image_id]]
                },
                "object" : {
                    "count" : 1,
                    "bbox" : {
                        "xmin" : [x1],
                        "xmax" : [x2],
                        "ymin" : [y1],
                        "ymax" : [y2],
                        "label" : [image_labels[image_id]],
                        "text": [new_label_to_original_label_map[image_labels[image_id]]]
                    },
                    "parts" : {
                        "x" : parts_x,
                        "y" : parts_y,
                        "v" : parts_v
                    },
                    "id" : [image_id],
                    "area" : [w * h]
                }
            })
    return train_data, test_data

def create_validation_split(train_data, fraction_per_class=0.1, shuffle=True):
    """
    Take `images_per_class` from the train dataset and create a validation set.
    """

    subset_train_data = []
    val_data = []
    val_label_counts = {}

    class_labels = [i['class']['label'] for i in train_data]
    images_per_class = Counter(class_labels)
    val_images_per_class = {label : 0 for label in images_per_class.keys()}

    # Sanity check to make sure each class has more than 1 label
    for label, image_count in images_per_class.items():
        if image_count <= 1:
            print("Warning: label %d has only %d images" % (label, image_count))

    if shuffle:
        random.shuffle(train_data)

    for image_data in train_data:
        label = image_data['class']['label']

        if label not in val_label_counts:
            val_label_counts[label] = 0

        if val_images_per_class[label] < images_per_class[label] * fraction_per_class:
            val_data.append(image_data)
            val_images_per_class[label] += 1
        else:
            subset_train_data.append(image_data)

    return subset_train_data, val_data
