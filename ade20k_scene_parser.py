import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle as pkl
from PIL import Image

# This is the name of the pkl file used to parse through the files.
DATASET_PATH = 'ADE20K_2021_17_01'
index_file = 'index_ade20k.pkl'
# Open the pkl file at dataset path and index file denoted above.
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)

nfiles = len(index_ade20k['filename'])

################################################################################
# The following are all helper functions for the function Parse Scenes, which
# produces training validation and test images to load based on the number of 
# classes (or scenes) we want to train our model on.

# Gets the total number of occurences of each scene and keeps track of what scenes
# are in the training and test sets (occurences = number of images in scene).
def get_scene_occurences():
    # Get the top num_scenes classes
    scene_occurence = {}
    training_scenes = set()
    test_scenes = set()
    for i in range(nfiles):
        scene = index_ade20k['scene'][i]
        scene = scene.split('/')
        if scene[0] == '':
            scene = scene[1]
        else:
            scene = scene[0]
        full_file_name = '{}/{}'.format(index_ade20k['folder'][i],
                                        index_ade20k['filename'][i])
        if scene not in scene_occurence.keys():
            scene_occurence[scene] = 1
        else:
            scene_occurence[scene] += 1
        if 'training' in full_file_name:
            training_scenes.add(scene)
        else:
            test_scenes.add(scene)
    return scene_occurence, training_scenes, test_scenes

# Given the number of classes (scenes we want to use), this function extracts
# the scenes to be used (takes num_classes most occuring scenes granted the
# scenes appear in both training and test sets).
def get_used_scenes(scene_occurence, training_scenes, test_scenes, num_classes):
    scene_sorted = dict(sorted(scene_occurence.items(), key=lambda item: item[1], reverse=True))
    used_scenes = set()
    image_sum = 0
    for k, v in scene_sorted.items():
        if k in test_scenes and k not in training_scenes:
            continue
        if k in training_scenes and k not in test_scenes:
            continue
        used_scenes.add(k)
        image_sum += v
        if len(used_scenes) == num_classes:
            break
    return used_scenes

# This function maps the scenes to indexes and indexes to scenes for classification
# purposes. The index_to_scene.pkl should be saved in the Datasets folder
# where you can extract the scene classification 
def map_scene_to_index(used_scenes):
    counter = 0
    scene_to_index = {}
    index_to_scene = {}
    for scene in used_scenes:
        if scene not in scene_to_index.keys():
            scene_to_index[scene] = counter
            index_to_scene[counter] = scene
            counter += 1
    file = open('index_to_scene.pkl', 'wb')
    pkl.dump(index_to_scene, file)
    file.close()
    return scene_to_index

# This function serves two purposes:
# 1. Creates a dictionary mapping scenes to the indices of all possible training
# examples (which the validation set will later extract from).

# 2. Creates an array of the indices of the test images to be used.
def get_total_train_indices(used_scenes):
    image_count = 0
    total_train = {}
    total_test = []
    train_count = 0
    for i in range(nfiles):
        # Get Scene
        scene = index_ade20k['scene'][i]
        scene = scene.split('/')
        if scene[0] == '':
            scene = scene[1]
        else:
            scene = scene[0]
        # Get file name to know if training or validation
        full_file_name = '{}/{}'.format(index_ade20k['folder'][i],
                                           index_ade20k['filename'][i])
        # If we're using scene.
        if scene in used_scenes:
            if 'training' in full_file_name:
                # This is a training set so add the index to the dictionary
                if scene in total_train.keys():
                    total_train[scene].append(i)
                else:
                    total_train[scene] = [i]
                train_count += 1
            # Test so just add index to list.
            else:
                total_test.append(i)
            image_count += 1
    return total_train, np.array(total_test, dtype=np.uint16), image_count, train_count

# This extracts the validation set indices and thus the training set indices
# from the dictionary mapping scenes to all possible training indices. 
def get_total_test_indices(total_train, image_count, val_size):
    train_indices = []
    validation_indices = []
    # This function assigns a proportionate amount of validation images
    # to pull from a scene given how many total images are there and the
    # expected size of the validation set.
    def num_images_to_pull(list_size):
        return int(np.floor((list_size / image_count) * val_size))
    for scene, image_list in total_train.items():
        num_val_images = num_images_to_pull(len(image_list))
        validation_indices += image_list[len(image_list)-num_val_images-1:]
        total_train[scene] = total_train[scene][:len(image_list)-num_val_images-1]
        train_indices += total_train[scene]
    return np.array(train_indices, dtype=np.uint16), np.array(validation_indices, dtype=np.uint16)


# This is the scene to call from other scenes. This combines all of the above
# helper functions to pull the training, test, and validation indices given
# the number of scenes to extract. It also saves a index_to_scene dictionary
# and returns a scene_to_index dictionary for classification purposes.
def parse_scenes(num_classes):
    data_dict = {}
    scene_occurence, training_scenes, test_scenes = get_scene_occurences()
    used_scenes = get_used_scenes(scene_occurence, training_scenes, test_scenes, num_classes)
    data_dict['scene_to_index'] = map_scene_to_index(used_scenes)
    total_train, data_dict['test'], image_count, train_count = get_total_train_indices(used_scenes)
    validation_size = .1 * train_count
    data_dict['train'], data_dict['val'] = get_total_test_indices(total_train, image_count, validation_size)
    assert(len(data_dict['train']) + len(data_dict['val']) == train_count and
           train_count + len(data_dict['test']) == image_count)
    return data_dict

################################################################################