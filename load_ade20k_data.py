# Note: This was ran locally. If issues with imports that is why.
import utils_ade20k
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle as pkl
from PIL import Image, ImageOps
from ade20k_scene_parser import parse_scenes

# This is the name of the pkl file used to parse through the files.
DATASET_PATH = 'ADE20K_2021_17_01'
index_file = 'index_ade20k.pkl'
# Open the pkl file at dataset path and index file denoted above.
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)

# Specify how many classes you want to classify here.
num_classes = 50
# This is from scene parser class to parse scenes.
data_indices = parse_scenes(num_classes)

print('Number of training examples: {}'.format(len(data_indices['train'])))
print('Number of validation examples: {}'.format(len(data_indices['val'])))
print('Number of test examples: {}'.format(len(data_indices['test'])))

# Make sure no indices in train, test, and validation are in any of the others.
for index in data_indices['train']:
    if index in data_indices['val']:
        print('WARNING: Overlapping training validation indices.')
    if index in data_indices['test']:
        print('WARNING: Overlapping test validation_indices.')

for index in data_indices['val']:
    if index in data_indices['test']:
        print('WARNING: Overlapping validation and test indices.')

# Given an array of indices with name variable_name, load the raw images into
# the array. Assuming the array is initialized to an np array of shape
# (len(indices), img_size, img_size, 3) where 3 is number of channels.
def load_raw_images(indices, variable_name, array):
    img_size = array.shape[1]
    print('Beginning {} load.'.format(variable_name))
    for counter, i in enumerate(indices):
        full_file_name = '{}/{}'.format(index_ade20k['folder'][i],
                                        index_ade20k['filename'][i])
        if full_file_name is None:
            print('Issue with image name at i = {}.'.format(i))
            continue
        img = cv2.imread(full_file_name)[:, :, ::-1]
        reshaped_image = cv2.resize(img, dsize=(img_size, img_size),
                                    interpolation=cv2.INTER_CUBIC)
        array[counter] = reshaped_image
        if (counter + 1) % 100 == 0:
            print('{} of {} images uploaded.'.format(counter + 1, len(indices)))
    print('{} upload complete, saving file....'.format(variable_name))
    np.save('{}.npy'.format(variable_name), array)
    print('{} save complete.'.format(variable_name))

# Given an array of indices with name variable_name, load the seg masks into
# the array. Assuming the array is initialized to an np array of shape
# (len(indices), img_size, img_size, 3) where 3 is number of channels.
def load_seg_masks(indices, variable_name, array):
    img_size = array.shape[1]
    print('Beginning {} load.'.format(variable_name))
    for counter, i in enumerate(indices):
        full_file_name = '{}/{}'.format(index_ade20k['folder'][i],
                                        index_ade20k['filename'][i])
        if full_file_name is None:
            print('Issue with image name at i = {}.'.format(i))
            continue
        fileseg = full_file_name.replace('.jpg', '_seg.png');
        seg = Image.open(fileseg)
        seg = np.array(ImageOps.grayscale(seg))

        reshaped_image = cv2.resize(seg, dsize=(img_size, img_size),
                                    interpolation=cv2.INTER_CUBIC)
        reshaped_image = reshaped_image.reshape(img_size, img_size, 1)
        array[counter] = reshaped_image
        if (counter + 1) % 100 == 0:
            print('{} of {} images uploaded.'.format(counter + 1, len(indices)))
    print('{} upload complete, saving file....'.format(variable_name))
    print(np.unique(array))
    np.save('{}.npy'.format(variable_name), array)
    print('{} save complete.'.format(variable_name))

# Given an array of indices and an array with name variable_name, labels all
# the images at specified indices with appropriate class based on scene parsing
# done from the above parse_scenes function call.
def load_scenes(indices, variable_name, array):
    print('Beginning {} load'.format(variable_name))
    for counter, i in enumerate(indices):
        full_file_name = '{}/{}'.format(index_ade20k['folder'][i],
                                        index_ade20k['filename'][i])
        if full_file_name is None:
            print('Issue with image name at i = {}.'.format(i))
            continue
        scene = index_ade20k['scene'][i]
        scene = scene.split('/')
        if scene[0] == '':
            scene = scene[1]
        else:
            scene = scene[0]
        index = data_indices['scene_to_index'][scene]
        array[counter] = index
    np.save('{}.npy'.format(variable_name), array)
    print('Upload and save complete for {}'.format(variable_name))

# Use above helper functions to save the x arrays.
def get_ade20k_x(train_indices, val_indices, test_indices):
    img_size = 128
    num_train = len(train_indices)
    num_val = len(val_indices)
    num_test = len(test_indices)
    # Training
    x_train = np.zeros((num_train, img_size, img_size, 3))
    load_raw_images(train_indices, 'x_train', x_train)
    # Set to None for RAM
    x_train = None

    # Validation
    x_val = np.zeros((num_val, img_size, img_size, 3))
    load_raw_images(val_indices, 'x_val', x_val)
    x_val = None

    # Test
    x_test = np.zeros((num_test, img_size, img_size, 3))
    load_raw_images(test_indices, 'x_test', x_test)
    x_test = None

# Use above helper functions to save the y_seg arrays.
def get_ade20k_seg(train_indices, val_indices, test_indices):
    img_size = 128
    num_train = len(train_indices)
    num_val = len(val_indices)
    num_test = len(test_indices)

    # Training
    y_train_seg = np.zeros((num_train, img_size, img_size, 1))
    load_seg_masks(train_indices, 'y_train_seg', y_train_seg)
    y_train_seg = None

    # Validation
    y_val_seg = np.zeros((num_val, img_size, img_size, 1))
    load_seg_masks(val_indices, 'y_val_seg', y_val_seg)
    y_val_seg = None

    # Test
    y_test_seg = np.zeros((num_test, img_size, img_size, 1))
    load_seg_masks(test_indices, 'y_test_seg', y_test_seg)
    y_test_seg = None

# Use the above helper functions to save the y_scene arrays.
def get_ade20k_scenes(train_indices, val_indices, test_indices):
    num_train = len(train_indices)
    num_val = len(val_indices)
    num_test = len(test_indices)

    y_train_scenes = np.zeros(num_train)
    load_scenes(train_indices, 'y_train_scenes', y_train_scenes)
    y_train_scenes = None

    y_val_scenes = np.zeros(num_val)
    load_scenes(val_indices, 'y_val_scenes', y_val_scenes)
    y_val_scenes = None

    y_test_scenes = np.zeros(num_test)
    load_scenes(test_indices, 'y_test_scenes', y_test_scenes)
    y_test_scenes = None

# Running below lines will save x, y_seg, and y_train arrays to you local hard
# drive or system path respectively.

# Run one at a time for RAM and Hardware Memory purposes (comment out other two).
get_ade20k_x(data_indices['train'], data_indices['val'], data_indices['test'])
get_ade20k_seg(data_indices['train'], data_indices['val'], data_indices['test'])
get_ade20k_scenes(data_indices['train'], data_indices['val'], data_indices['test'])