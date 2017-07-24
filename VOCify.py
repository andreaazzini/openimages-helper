import csv
import os
import urllib

BASE_DIR = os.path.join('/', 'datasets', 'OpenImages')
OPEN_IMAGES_DIR = os.path.join(BASE_DIR, '2017_07')
download_enabled = True
trainval_enabled = True

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
        return dict(zip(label_names, classes))

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

def count_labels(dataset_partition, label_names):
    label_count = {k: 0 for k in label_names}
    annotations_filename = os.path.join(OPEN_IMAGES_DIR, dataset_partition, 'annotations-human-bbox.csv')
    with open(annotations_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            label_name = row['LabelName']
            if label_name in label_names:
                label_count[label_name] += 1
    return label_count

def download_images(dataset_partition, image_ids):
    images_filename = os.path.join(OPEN_IMAGES_DIR, dataset_partition, 'images.csv')
    images_dir = os.path.join(BASE_DIR, 'VOCify', 'JPEGImages')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    with open(images_filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            image_id = row['ImageID']
            if image_id in image_ids:
                image_filename = "%s.jpg" % image_id
                image_path = os.path.join(images_dir, image_filename)
                image_url = row['OriginalURL']
                print('Getting %s, storing in %s' % (image_url, image_path))
                urllib.urlretrieve(image_url, image_path)

def write_image_sets(dataset_partition, images):
    image_sets_dir = os.path.join(BASE_DIR, 'VOCify', 'ImageSets', 'Main')
    if not os.path.exists(image_sets_dir):
        os.makedirs(image_sets_dir)
    image_set_path = os.path.join(image_sets_dir, '%s.txt' % dataset_partition)

    with open(image_set_path, 'w') as f:
        for image in images:
            f.write('%s\n' % image)

def write_annotations(images, format='voc'):
    import xml.etree.cElementTree as ET

    annotations_dir = os.path.join(BASE_DIR, 'VOCify', 'Annotations')
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    for image in images:
        annotation = ET.Element('annotation')
        ET.SubElement(annotation, 'filename').text = '%s.jpg' % image
        ET.SubElement(annotation, 'folder').text = '2017_07'
        for label in images[image]:
            for xmin, xmax, ymin, ymax in images[image][label]:
                obj = ET.SubElement(annotation, 'object')
                ET.SubElement(obj, 'name').text = label
                bndbox = ET.SubElement(obj, 'bndbox')
                ET.SubElement(bndbox, 'xmin').text = xmin
                ET.SubElement(bndbox, 'xmax').text = xmax
                ET.SubElement(bndbox, 'ymin').text = ymin
                ET.SubElement(bndbox, 'ymax').text = ymax

        annotation_path = os.path.join(annotations_dir, '%s.xml' % image)
        tree = ET.ElementTree(annotation)
        tree.write(annotation_path)

if __name__ == '__main__':
    dataset = {}
    label_dict = build_label_dict()

    for dataset_partition in ['train', 'validation', 'test']:
        dataset[dataset_partition] = get_images_with_annotations(dataset_partition, label_dict.keys())
        print('Found %d %s images. Saving...' % (len(dataset[dataset_partition]), dataset_partition))
        if download_enabled:
            print('Downloading %s images...' % dataset_partition)
            download_images(dataset_partition, dataset[dataset_partition].keys())
        write_image_sets(dataset_partition, dataset[dataset_partition].keys())
        print('Writing annotations for the %s images...' % dataset_partition)
        write_annotations(dataset[dataset_partition])

    if trainval_enabled:
        print("Trainval enabled, writing trainval.txt...")
        write_image_sets('trainval', dataset['train'].keys() + dataset['validation'].keys())

    print('Done!')
