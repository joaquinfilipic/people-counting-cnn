# -------------------------------------------------------------------------------------------------
# This script performs the testing of an already trained neural network agains a set of images that
# were not used in the training nor the validating stages of it.
# -------------------------------------------------------------------------------------------------

import argparse
import h5py 
import torch
import numpy as np
import json
import os
import glob

from PIL import Image
from torchvision import transforms, models
from model import UNet
from library import merge_matrix_values, extract_prediction_positions, generate_label, join_rgb_ir

# -------------------------------------------------------------------------------------------------

# args
parser = argparse.ArgumentParser()
parser.add_argument('--trained_network', action = 'store', type = str, required = True)
parser.add_argument('--threshold', action = 'store', type = float, default = 30)
parser.add_argument('--ir_enabled', action = 'store', type = bool, default = False)
parser.add_argument('--images_dir', action = 'store', type = str, required = True)
parser.add_argument('--labels_dir', action = 'store', type = str, required = True)
parser.add_argument('--positions_dir', action = 'store', type = str, required = True)
parser.add_argument('--predictions_dir', action = 'store', type = str, required = True)
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
images_paths = []
for image_path in glob.glob(os.path.join(rgb_images_directory, '*.jpg')):
    images_paths.append(image_path)

analyzed_images = 0
remaining = len(images_paths)

# iterate for every testing file
for image_path in images_paths:

    print('Analyzed images: {}. Remaining: {}.'.format(analyzed_images, remaining))
    print('Analyzing image: {}.'.format(image_path))

    # get input image
    rgb_image_path = image_path
    ir_image_path = image_path.replace('rgb', 'ir')
    if args.ir_enabled:
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

    # get label information
    label_path = image_path.replace(rgb_images_directory, args.labels_dir).replace('rgb_image', 'label').replace('.jpg', '.h5')
    label_file = h5py.File(label_path, mode = 'r')
    label = np.asarray(label_file['label'])
    true_count = np.asarray(label_file['count'])

    # get positions
    positions_path = image_path.replace(rgb_images_directory, args.positions_dir).replace('rgb_image', 'positions').replace('.jpg', '.h5')
    positions_file = h5py.File(positions_path, mode = 'r')
    true_positions = np.asarray(positions_file['positions'])

    # extract predicted positions using given threshold
    processed_matrix = merge_matrix_values(prediction_matrix)
    predicted_positions = extract_prediction_positions(processed_matrix, args.threshold)

    # save prediction
    if args.ir_enabled:
        prediction_path = image_path.replace(rgb_images_directory, args.predictions_dir + '/rgb-ir').replace('rgb_image', 'prediction').replace('.jpg', '.h5')
    else:
        prediction_path = image_path.replace(rgb_images_directory, args.predictions_dir + '/rgb').replace('rgb_image', 'prediction').replace('.jpg', '.h5')
    with h5py.File(prediction_path, 'w') as hf:
        hf['prediction'] = prediction_matrix
        hf['predicted_positions'] = predicted_positions
        hf['true_positions'] = true_positions
        hf['predicted_count'] = len(predicted_positions)
        hf['true_count'] = true_count

    print('predicted_count/true_count: [{}/{}].'.format(len(predicted_positions), true_count))

    analyzed_images = analyzed_images + 1
    remaining = remaining - 1

print('All images analyzed successfully.')