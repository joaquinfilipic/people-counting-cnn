# -------------------------------------------------------------------------------------------------
# This script performs different testing and results analysis in order to find the optimum
# threshold value.
# -------------------------------------------------------------------------------------------------

import argparse
import h5py 
import torch
import numpy as np
import json
import os
import glob
import math

from statistics import mean
from PIL import Image
from torchvision import transforms, models
from model import UNet
from library import merge_matrix_values, extract_prediction_positions, join_rgb_ir

# tests an image with a trained network for all given thresholds in an array
def get_threshold_errors_for_image(network, ir_enabled, image_path, thresholds, images_dir, labels_dir):

    image_threshold_errors = []

    # get input image
    rgb_image_path = image_path
    ir_image_path = image_path.replace('rgb', 'ir')
    if ir_enabled:
        network_input = join_rgb_ir(rgb_image_path, ir_image_path)
    else:
        network_input = Image.open(rgb_image_path).convert('RGB')

    # transform input image to tensor and load to cuda
    network_input = transform(network_input).cuda()

    # pass image as tensor through network
    output = network(network_input.unsqueeze(0))

    output_matrix = output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3])

    # convert tensor object to matrix and truncate to int
    prediction_matrix = np.asarray(output_matrix)
    prediction_matrix = prediction_matrix.astype(int)

    # get true_count from label
    label_path = image_path.replace(images_dir, labels_dir).replace('rgb_image', 'label').replace('.jpg', '.h5')
    label_file = h5py.File(label_path, mode = 'r')
    true_count = np.asarray(label_file['count'])

    # merge matrix values and get locale maximums
    processed_matrix = merge_matrix_values(prediction_matrix)

    # extract predicted positions for all thresholds
    for threshold in thresholds:

        predicted_positions = extract_prediction_positions(processed_matrix, threshold)
        predicted_count = len(predicted_positions)
        count_error = abs(predicted_count - true_count)
        relative_count_error = count_error / true_count

        # append relative count error for current threshold to an array of threshold errors
        image_threshold_errors.append(relative_count_error)
    
    return image_threshold_errors

# -------------------------------------------------------------------------------------------------

# args
parser = argparse.ArgumentParser()
parser.add_argument('--trained_network', action = 'store', type = str, required = True)
parser.add_argument('--ir_enabled', action = 'store', type = bool, default = False)
parser.add_argument('--threshold_min_value', action = 'store', type = float, required = True)
parser.add_argument('--threshold_max_value', action = 'store', type = float, required = True)
parser.add_argument('--step_size', action = 'store', type = float, required = True)
parser.add_argument('--images_dir', action = 'store', type = str, required = True)
parser.add_argument('--labels_dir', action = 'store', type = str, required = True)
args = parser.parse_args()

# print configuration
print('configuration: {}.'.format(args))

rgb_images_directory = args.images_dir + '/rgb'

# network model
network = UNet(input_filters = 4) if args.ir_enabled else UNet()
network = network.cuda()

# load trained network
checkpoint = torch.load(args.trained_network)
network.load_state_dict(checkpoint['state_dict'])

transform = transforms.ToTensor()

# find all testing images paths
image_paths = []
for image_path in glob.glob(os.path.join(rgb_images_directory, '*.jpg')):
    image_paths.append(image_path)

# get all thresholds values to analize
thresholds = []
current_threshold = args.threshold_min_value
while current_threshold <= args.threshold_max_value:
    thresholds.append(current_threshold)
    current_threshold += args.step_size

analyzed_images = 0
remaining = len(image_paths)

threshold_errors = []

for image_path in image_paths:

    print('Analyzed images: {}. Remaining: {}.'.format(analyzed_images, remaining))
    print('Next image: {}.'.format(image_path))

    errors = get_threshold_errors_for_image(network, args.ir_enabled, image_path, thresholds, rgb_images_directory, args.labels_dir)
    threshold_errors.append(errors)

    analyzed_images += 1
    remaining -= 1

print("threshold, mean_error")

best_threshold = thresholds[0]
best_mean_error = math.inf

for idx in range(0, len(thresholds)):

    images_threshold_errors = []
    current_threshold = thresholds[idx]
    for errors in threshold_errors:
        images_threshold_errors.append(errors[idx])
    mean_error = mean(images_threshold_errors)

    print('{}, {}'.format(current_threshold, mean_error))

    if mean_error < best_mean_error:
        best_mean_error = mean_error
        best_threshold = current_threshold

print('best threshold value: {} with mean error of {}.'.format(best_threshold, best_mean_error))