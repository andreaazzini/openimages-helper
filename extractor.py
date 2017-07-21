import csv
import os
import urllib

BASE_DIR = os.path.join('/', 'datasets', 'OpenImages')
OPEN_IMAGES_DIR = os.path.join(BASE_DIR, '2017_07')

def get_label_name(label_description):
    class_description_filename = os.path.join(OPEN_IMAGES_DIR, 'class-descriptions.csv')
    with open(class_description_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=['label_name', 'label_description'])
        for row in reader:
            if row['label_description'] == label_description:
                return row['label_name']

def build_label_dict():
    with open('classes.txt', 'r') as classesfile:
        classes = classesfile.read().splitlines()
        classes = map(lambda c: c.capitalize(), classes)
        label_names = map(get_label_name, classes)
        return dict(zip(classes, label_names))

def get_images_with_annotations(dataset_partition, label_names):
    images = {}
    annotations_filename = os.path.join(OPEN_IMAGES_DIR, dataset_partition, 'annotations-human-bbox.csv')
    with open(annotations_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label_name = row['LabelName']
            if label_name in label_names:
                image_id = row['ImageID']
                bbox = (row['XMin'], row['XMax'], row['YMin'], row['YMax'])
                if image_id in images:
                    if label_name in images[image_id]:
                        images[image_id][label_name].append(bbox)
                    else:
                        images[image_id][label_name] = [bbox]
                else:
                    images[image_id] = {label_name: [bbox]}
    return images

def download_images(dataset_partition, image_ids):
    images_filename = os.path.join(OPEN_IMAGES_DIR, dataset_partition, 'images.csv')
    destination_dir = os.path.join(BASE_DIR, 'extractor', dataset_partition)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    with open(images_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_id = row['ImageID']
            if image_id in image_ids:
                image_filename = "%s.jpg" % image_id
                image_path = os.path.join(destination_dir, image_filename)
                image_url = row['OriginalURL']
                print('Getting %s, storing in %s' % (image_url, image_path))
                urllib.urlretrieve(image_url, image_path)

label_dict = build_label_dict()
print('Labels: %s' % str(label_dict.keys()))
train_images = get_images_with_annotations('train', label_dict.values())
print('Found %d train images' % len(train_images))
validation_images = get_images_with_annotations('validation', label_dict.values())
print('Found %d validation images' % len(validation_images))
test_images = get_images_with_annotations('test', label_dict.values())
print('Found %d test images' % len(test_images))

#print('Downloading train images...')
#download_images('train', train_images.keys())
#print('Downloading validation images...')
#download_images('validation', validation_images.keys())
#print('Downloading test images...')
#download_images('test', test_images.keys())
