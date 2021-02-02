# This script creates image path sets for training and validating the neural network.
# Two files are created, containing the set of paths for the RGB images. This is because the IR 
# images can be obtained later by replacing 'rgb' with 'ir' in each file.
# Notation:
#     training:   used in the training of the neural network to adjust the weights en each epoch.
#     validating: used after each epoch of the neural network's training to validate that it's
# 				  improving regarding images not used in the training.

import argparse
import os
import glob
import json
import random

# -------------------------------------------------------------------------------------------------

# args
parser = argparse.ArgumentParser()
parser.add_argument('--images_dir', action = 'store', type = str, required = True)
parser.add_argument('--training_file', action = 'store', type = str, required = True)
parser.add_argument('--validating_file', action = 'store', type = str, required = True)
parser.add_argument('--validating_count', action = 'store', type = int, required = True)
args = parser.parse_args()

# print configuration
print('configuration: {}.'.format(args))

# define location of images directory
rgb_images_directory = args.images_dir + '/rgb'

all_rgb_image_paths = []

# get all image paths in an array
for rgb_image_path in glob.glob(os.path.join(rgb_images_directory, '*.jpg')):
	all_rgb_image_paths.append(rgb_image_path)

# we shuffle the paths
random.shuffle(all_rgb_image_paths)

# training starts with all the image paths
training_rgb_paths = all_rgb_image_paths

# validating array starts empty
validating_rgb_paths = []

# populate validating sets by removing a given number of files from training set
for count in range(0, args.validating_count):
	# move first path from training to validating
	path = training_rgb_paths.pop(0)
	print('Adding image {} to validating set.'.format(path))
	validating_rgb_paths.append(path)

# save path sets to files
with open(args.training_file, 'w') as training_outfile:
	json.dump(training_rgb_paths, training_outfile)
with open(args.validating_file, 'w') as validating_outfile:
	json.dump(validating_rgb_paths, validating_outfile)

print('Filename dump completed.')