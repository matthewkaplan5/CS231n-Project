import numpy as np
# Script: get_ade20k_data.py

# This script loads the presaved train, test, and validation data to be used.
# View ade20k_scene_parser.py and load_ade20k_data.py for the two scripts
# that pull the training, testing, and validation data from the dataset
# based on the number of classes (scenes) we want to use for classification.

# Note: This script preprocesses the data by subtracting the training mean from 
# each of the splits.

# This function returns a dictionary containing all the different data arrays.
# Argument is a path to the folder where the arrays currently stored.
# Assumes arrays are .npy files that are preprocessed using ade20k_scene_parser.py
# and load_ade20k_data.py scripts.
def get_ade20k_data(path):
  if path[len(path) - 1] != '/':
    path = path + '/'
  data = {}

  # Raw Images (x)
  data['x_train'] = np.load(path + 'x_train.npy')
  data['x_val'] = np.load(path + 'x_val.npy')
  data['x_test'] = np.load(path + 'x_test.npy')

  # Subtract Mean
  # x_mean = np.mean(data['x_train'])
  # data['x_train'] -= x_mean
  # data['x_val'] -= x_mean
  # data['x_test'] -= x_mean

  # Segmentation Masks (y_seg)
  data['y_train_seg'] = np.load(path + 'y_train_seg.npy')
  data['y_val_seg'] = np.load(path + 'y_val_seg.npy')
  data['y_test_seg'] = np.load(path + 'y_test_seg.npy')

  # Subtract Mean
  # y_seg_mean = np.mean(data['y_train_seg'])
  # data['y_train_seg'] -= y_seg_mean
  # data['y_val_seg'] -= y_seg_mean
  # data['y_test_seg'] -= y_seg_mean

  # Scene labels
  data['y_train_scenes'] = np.load(path + 'y_train_scenes.npy')
  data['y_val_scenes'] = np.load(path + 'y_val_scenes.npy')
  data['y_test_scenes'] = np.load(path + 'y_test_scenes.npy')

  # Subtract Mean
  # y_scene_mean = np.mean(data['y_train_scenes'])
  # data['y_train_scenes'] -= y_scene_mean
  # data['y_val_scenes'] -= y_scene_mean
  # data['y_test_scenes'] -= y_scene_mean
  
  return data

