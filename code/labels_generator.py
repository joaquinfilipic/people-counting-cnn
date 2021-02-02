import argparse
import h5py
import scipy.io as io
import numpy as np
import os
import glob

from matplotlib import pyplot as plt
from library import generate_label

# -------------------------------------------------------------------------------------------------

# args
parser = argparse.ArgumentParser()
parser.add_argument('--positions_dir', action = 'store', type = str, required = True)
parser.add_argument('--images_dir', action = 'store', type = str, required = True)
parser.add_argument('--labels_dir', action = 'store', type = str, required = True)
args = parser.parse_args()

positions_directory = args.positions_dir
rgb_images_directory = args.images_dir + '/rgb'
labels_directory = args.labels_dir

# find all positions files in given directory
positions_files = []
for positions_file_path in glob.glob(os.path.join(positions_directory, '*.h5')):
    positions_files.append(positions_file_path)

generated_labels = 0
remaining = len(positions_files)

# generate labels and save them as .h5 files
for positions_file_path in positions_files:

	print('Generated labels: {}. Remaining: {}.'.format(generated_labels, remaining))
	print('Processing file: {}.'.format(positions_file_path))

	# get array of positions
	h5_file = h5py.File(positions_file_path, 'r')
	positions = np.asarray(h5_file['positions'])

	# get image for current positions file to extract shape
	image_file_path = positions_file_path.replace(positions_directory, rgb_images_directory).replace('positions', 'rgb_image').replace('.h5', '.jpg')
	image = plt.imread(image_file_path)

	# generate and save label
	label = generate_label(positions, image.shape)
	label_file_path = positions_file_path.replace(positions_directory, labels_directory).replace('positions', 'label')
	with h5py.File(label_file_path, 'w') as label_h5_file:
		label_h5_file['label'] = label
		label_h5_file['count'] = len(positions)

	generated_labels = generated_labels + 1
	remaining = remaining - 1

print('Label generation finished.')