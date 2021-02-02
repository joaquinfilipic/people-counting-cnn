# -------------------------------------------------------------------------------------------------
# This script performs the analysis of all the predictions found in a given directory. This 
# analysis consists in counting the sum of relative errors between the predicted count of people
# and the actual count.
# -------------------------------------------------------------------------------------------------

import argparse
import h5py 
import torch
import numpy as np
import os
import glob
import json

from statistics import mean, stdev
from datetime import datetime
from library import find_position_errors

# -------------------------------------------------------------------------------------------------

# Finds the translation value (to convert pixels to meters) for a given path. This is done by
# searching for one of the keys received in the dictionary.
def find_translation(dictionary, path):

    for i in range(0, len(dictionary)):
        search = dictionary[i]["key"] + '.'
        if search in path:
            return dictionary[i]["value"]

# -------------------------------------------------------------------------------------------------

# Translates pixels to cm. This is done by finding the correct translation to the height of the
# image in the given directory. Then, applies this translation to all the values in the array.
def translate_pixels_to_cm(distances, dictionary, path):

    cm_in_pixel = find_translation(dictionary, path)
    
    cm_distances = []
    for distance_in_pixels in distances:
        cm_distances.append(distance_in_pixels * cm_in_pixel)

    return cm_distances

# -------------------------------------------------------------------------------------------------

# args
parser = argparse.ArgumentParser()
parser.add_argument('--predictions_dir', action = 'store', type = str, required = True)
parser.add_argument('--analyze_positions_error', action = 'store', type = bool, required = False, default = False)
parser.add_argument('--dictionary', action = 'store', type = str, required = True)
parser.add_argument('--start_number', action = 'store', type = int, required = False)
args = parser.parse_args()

# print configuration
print('configuration: {}.'.format(args))

# load dictionary
dictionary = json.loads(args.dictionary)

# find all prediction files in given directory
prediction_files = []
for prediction_file_path in glob.glob(os.path.join(args.predictions_dir, '*.h5')):
    prediction_files.append(prediction_file_path)
sorted_files = prediction_files.sort()

# initialize arrays to hold errors
relative_count_errors = []
mean_relative_positions_errors = []
stdev_relative_positions_errors = []

print('image_number, relative_count_error, position_errors_mean, position_errors_stdev')

start_number = 1
if args.start_number:
    start_number = args.start_number

# initiate analysis for each prediction file
for image_number in range(start_number, len(prediction_files) + 1):

    prediction_file = prediction_files[image_number - 1]

    # get prediction objetcs
    h5_file = h5py.File(prediction_file, 'r')
    predicted_positions = np.asarray(h5_file['predicted_positions'])
    true_positions = np.asarray(h5_file['true_positions'])
    predicted_count = np.asarray(h5_file['predicted_count'])
    true_count = np.asarray(h5_file['true_count'])

    # analysing difference between predicted count and the true count
    count_error = abs(predicted_count - true_count)
    relative_count_error = count_error / true_count

    # this is configurable because it requires a huge amount of time
    if args.analyze_positions_error and relative_count_error < 1.0:

        # analysing distance between predicted positions and the true ones
        relative_positions_errors = find_position_errors(predicted_positions, true_positions)

        # translate positions errors from pixels to cm
        relative_positions_errors = translate_pixels_to_cm(relative_positions_errors, dictionary, prediction_file)

        mean_relative_positions_error = mean(relative_positions_errors)
        stdev_relative_positions_error = stdev(relative_positions_errors)

        print('{}, {}, {}, {}'.format(image_number, relative_count_error, mean_relative_positions_error, stdev_relative_positions_error))

    else:
        
        print('{}, {}'.format(image_number, relative_count_error))

print('Analysis finished.')