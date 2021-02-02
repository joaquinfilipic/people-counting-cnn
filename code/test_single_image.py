# -------------------------------------------------------------------------------------------------
# This script performs the testing of an already trained neural network agains a single image.
# -------------------------------------------------------------------------------------------------

import argparse
import h5py 
import torch
import numpy as np
import json

from PIL import Image
from torchvision import transforms, models
from model import UNet
from library import extract_prediction_positions, join_rgb_ir, merge_matrix_values, generate_density_map
from pandas import *

# -------------------------------------------------------------------------------------------------

# args
parser = argparse.ArgumentParser()
parser.add_argument('--trained_network', action = 'store', type = str, required = True)
parser.add_argument('--rgb_image_path', action = 'store', type = str, required = True)
parser.add_argument('--threshold', action = 'store', type = float, default = 80)
parser.add_argument('--ir_enabled', action = 'store', type = bool, default = False)
parser.add_argument('--positions_file', action = 'store', type = str, required = True)
args = parser.parse_args()

# print configuration
print('configuration: {}.'.format(args))

# network model
network = UNet(input_filters = 4) if args.ir_enabled else UNet()
network = network.cuda()

# load trained network
checkpoint = torch.load(args.trained_network)
network.load_state_dict(checkpoint['state_dict'])

transform = transforms.ToTensor()

print('analyzing image: {}.'.format(args.rgb_image_path))

# get input image
rgb_image_path = args.rgb_image_path
ir_image_path = rgb_image_path.replace('rgb', 'ir')
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

# extract predicted positions using given threshold
processed_matrix = merge_matrix_values(prediction_matrix)
predicted_positions = extract_prediction_positions(processed_matrix, args.threshold)

print('predicted_count: {}.'.format(len(predicted_positions)))

with h5py.File(args.positions_file, 'w') as hf:
    hf['positions'] = predicted_positions