# This file is intended to be a library containing multiple functions needed by the rest of the
# project.

import h5py
import numpy as np
import scipy
import scipy.io as io
import random
import math
import torch
import torchvision.transforms.functional as F
import shutil
import time

from scipy import spatial
from PIL import Image
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset as Dataset
from torchvision import transforms, models
from matplotlib import pyplot as plt
from functools import cmp_to_key
from datetime import datetime
from copy import copy, deepcopy

# -------------------------------------------------------------------------------------------------

# Network utils
def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
            
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best, checkpoint_filename):
    torch.save(state, checkpoint_filename)
    if is_best:
        shutil.copyfile(checkpoint_filename, 'best_' + checkpoint_filename)

# -------------------------------------------------------------------------------------------------

# Loads and joins the given rgb and ir images in one single matrix
def join_rgb_ir(rgb_image_path, ir_image_path):

    # get rgb and ir images
    rgb_image = plt.imread(rgb_image_path)
    ir_image = plt.imread(ir_image_path)

    # initialize matrix to join rgb with ir data
    result_image = np.zeros((rgb_image.shape[0], rgb_image.shape[1], 4), dtype = np.float32)

    # populate matrix
    for y in range(0, rgb_image.shape[0]):
        for x in range(0, rgb_image.shape[1]):
            # set RED
            result_image[y][x][0] = rgb_image[y][x][0]
            # set GREEN
            result_image[y][x][1] = rgb_image[y][x][1]
            # set BLUE
            result_image[y][x][2] = rgb_image[y][x][2]
            # set IR
            if len(ir_image.shape) == 3:
                result_image[y][x][3] = ir_image[y][x][0]
            else:
                result_image[y][x][3] = ir_image[y][x]

    return result_image

# -------------------------------------------------------------------------------------------------

# Loads and returns an image and its corresponding label. For this, uses the given image path.
def load_data(images_dir, labels_dir, rgb_image_path, ir_enabled):

    rgb_image_dir = images_dir + '/rgb'

    # obtains path for the label asociated with the given image path
    label_path = rgb_image_path.replace(rgb_image_dir, labels_dir).replace('rgb_image', 'label').replace('.jpg', '.h5')

    # open label file and extract its content as an array to return
    label_file = h5py.File(label_path, mode = 'r')
    target = np.asarray(label_file['label'])

    if ir_enabled:
        # load and join rgb and ir images
        network_input_image = join_rgb_ir(rgb_image_path, rgb_image_path.replace('rgb', 'ir'))

        return network_input_image, target

    else:
        # get rgb image only
        network_input_image = Image.open(rgb_image_path).convert('RGB')
    
        return network_input_image, target

# -------------------------------------------------------------------------------------------------

# Generates a matrix with all positions in 0 and 100 in the given points array.
#     points: array of integer pairs.
#     shape: shape of the output label
def generate_label(points, shape):

    # prepare label as matrix of zeros using given shape
    label = np.zeros((shape[0], shape[1]))

    # insert value of 100 in points positions
    for point in points:
        x = int(round(point[0]))
        y = int(round(point[1]))
        if y >= shape[0] or x >= shape[1]:
            print('ERROR: found point: {} with value greater than shape: {}.'.format(point, shape))
        else:
            label[y][x] = 100

    return label

# -------------------------------------------------------------------------------------------------
                
# Function that receives a matrix, a point of that matrix and checks its neighbors. If the main
# point and a neighbor have values greater than 0, the smaller value is merged with the greatest of
# them. A submatrix of 3x3 is used.
def join_with_neighbors(matrix, point):

    height = matrix.shape[0]
    width = matrix.shape[1]

    point_value = matrix[point[1]][point[0]]

    if point_value == 0:
        return

    max_neighbor_value = 0

    # iterate over the submatrix of 3x3 around the main point
    for i in range(-1, 2):
        for j in range(-1, 2):

            if i == 0 and j == 0:
                continue

            x = i + point[0]
            y = j + point[1]

            # validate that the current point is within the boundaries
            if x >= 0 and x < width and y >= 0 and y < height and matrix[y][x] > 0:
                
                neighbor_value = matrix[y][x]

                # update max neighbor value and point        
                if neighbor_value > max_neighbor_value:
                    max_neighbor_value = neighbor_value
                    max_neighbor_point = (x,y)

    # if max neighbor is greater than current point, merge with it
    if max_neighbor_value > 0 and max_neighbor_value >= point_value:

        matrix[max_neighbor_point[1]][max_neighbor_point[0]] = max_neighbor_value + point_value
        matrix[point[1]][point[0]] = 0

# -------------------------------------------------------------------------------------------------

# Merges values of a matrix leaving only locale maximums.
def merge_matrix_values(input_matrix):

    # assert type of given matrix to int (to get rid of background noise)
    matrix = deepcopy(input_matrix)
    matrix = matrix.astype(int)

    # merge values representing posible positions
    for y in range(0, matrix.shape[0]):
        for x in range(0, matrix.shape[1]):
            join_with_neighbors(matrix, (x,y))

    return matrix

# -------------------------------------------------------------------------------------------------
                
# Extracts a list of positions with values greater than a threshold in the given matrix. 
#     input_matrix: matrix with only locale maximums.
#     threshold:    get all positions of values above this threshold.
# Returns an array with the positions extracted.
def extract_prediction_positions(matrix, threshold):

    # initialize prediction_positions list
    positions = []

    # generate positions
    for y in range(0, matrix.shape[0]):
        for x in range(0, matrix.shape[1]):
            if (matrix[y][x] >= threshold):
                positions.append((x,y))

    return positions

# -------------------------------------------------------------------------------------------------

# Calculates the distance between two points
def distance(point_1, point_2):

    x_distance = point_2[0] - point_1[0]
    y_distance = point_2[1] - point_1[1]

    return math.hypot(x_distance, y_distance)

# -------------------------------------------------------------------------------------------------

# Transforms a 2d array with dimensions 2*N in a list of pairs.
def two_dimension_array_to_pairs_list(matrix):

    response = []
    for i in range(0, matrix.shape[0]):
        response.append((matrix[i][0], matrix[i][1]))

    return response

# -------------------------------------------------------------------------------------------------

# Compares two pairs of positions to determine which has the closest distance between its positions
def positions_pairs_compare(pair_a, pair_b):

    distance_a = distance(pair_a[0], pair_a[1])
    distance_b = distance(pair_b[0], pair_b[1])

    if distance_a < distance_b:
        return -1
    if distance_a > distance_b:
        return 1

    a_first_distance_to_origin = distance(pair_a[0], (0,0))
    b_first_distance_to_origin = distance(pair_b[0], (0,0))

    if a_first_distance_to_origin < b_first_distance_to_origin:
        return -1
    if a_first_distance_to_origin > b_first_distance_to_origin:
        return 1

    a_second_distance_to_origin = distance(pair_a[1], (0,0))
    b_second_distance_to_origin = distance(pair_b[1], (0,0))

    if a_second_distance_to_origin < b_second_distance_to_origin:
        return -1
    if a_second_distance_to_origin > b_second_distance_to_origin:
        return 1

    return 0

# -------------------------------------------------------------------------------------------------

# Determines if two pairs of positions intersect. This happends when the first position of each
# pair are equals or if this is the case of the second position of each pair. 
def pairs_intersect(pair_a, pair_b):

    if pair_a[0] == pair_b[0] or pair_a[1] == pair_b[1]:
        return True
    
    return False

# -------------------------------------------------------------------------------------------------

# Determines if any of the pairs in the given array intersects with the potential match.
def any_intersects(pairs, potential_match):

    for pair in pairs:
        if pairs_intersect(potential_match, pair):
            return True
    
    return False

# -------------------------------------------------------------------------------------------------

def update_remaining_pairs(matches, consumer_list, suplier_list):

    remaining_consumer_list = []
    for consumer_position in consumer_list:
        found = False
        for match in matches:
            if consumer_position == match[0]:
                found = True
        if not found:
            remaining_consumer_list.append(consumer_position)
    
    remaining_suplier_list = []
    for suplier_position in suplier_list:
        found = False
        for match in matches:
            if suplier_position == match[1]:
                found = True
        if not found:
            remaining_suplier_list.append(suplier_position)

    pairs = cross_and_sort_pairs(remaining_consumer_list, remaining_suplier_list)

    return pairs, consumer_list, suplier_list

# -------------------------------------------------------------------------------------------------

# This function finds the N pairs with the shortest distance between their points in the given
# array.
def find_matches(pairs, count, consumer_list, suplier_list):

    matches = []
    for i in range(0, count):

        initial_time = time.time()
        match_found = False
        idx = 0
        while idx < len(pairs) and not match_found:

            if time.time() - initial_time > 29:
                pairs, consumer_list, suplier_list = update_remaining_pairs(matches, consumer_list, suplier_list)

            potential_match = pairs.pop(0)
            if not any_intersects(matches, potential_match):
                match_found = True

            idx += 1

        matches.append(potential_match)
    
    return matches

# -------------------------------------------------------------------------------------------------

def cross_and_sort_pairs(consumer_list, suplier_list):

    pairs = []
    for consumer_position in consumer_list:
        for suplier_position in suplier_list:
            pairs.append((consumer_position, suplier_position))
    
    # sort the pairs by distance between their points (ASC)
    sorted_pairs = sorted(pairs, key = cmp_to_key(positions_pairs_compare))

    return sorted_pairs

# -------------------------------------------------------------------------------------------------

# Finds the error between the predicted positions and the true ones. This is done by first finding
# the correct matches and then calculates the error of each match. Finally, returns a list with all
# those errors.
def find_position_errors(predicted_positions, true_positions):

    predicted_list = two_dimension_array_to_pairs_list(predicted_positions)
    true_list = two_dimension_array_to_pairs_list(true_positions)

    if (len(predicted_list) < len(true_list)):
        count = len(predicted_list)
        consumer_list = predicted_list
        suplier_list = true_list
    else:
        count = len(true_list)
        consumer_list = true_list
        suplier_list = predicted_list

    pairs = cross_and_sort_pairs(consumer_list, suplier_list)

    matches = find_matches(pairs, count, consumer_list, suplier_list)

    errors = []
    for match in matches:
        errors.append(distance(match[0], match[1]))

    return errors

# -------------------------------------------------------------------------------------------------

# This function receives 2 matrices and compares them calculating the difference of every pair of
# points between them. Finally, returns the mean square error of the comparation.
def calculate_matrix_mse(matrix_a, matrix_b):

    total_error = 0.0

    for y in range(0, matrix_a.shape[0]):
        for x in range(0, matrix_a.shape[1]):
            pixel_error = abs(matrix_a[y][x] - matrix_b[y][x])
            pixel_error = pow(pixel_error, 2)
            total_error += pixel_error

    return total_error / (matrix_a.shape[0] * matrix_a.shape[1])

# -------------------------------------------------------------------------------------------------

def generate_density_map(points, shape, k, pixel_to_cm):
  
    # initialize density matrix
    density = np.zeros(shape, dtype = np.float32)

    # for empty gt
    if len(points) == 0:
        return density

    pts = np.asarray(points)
  
    # create kdtree
    tree = spatial.KDTree(pts.copy(), leafsize = 2048)

    m, n = np.mgrid[0:shape[0], 0:shape[1]]
    all_points = list(zip(n.ravel(), m.ravel()))

    for point in all_points:
        x = point[0]
        y = point[1]

        # distances to the k nearest neighbors
        distances, _ = tree.query(point, k = k)

        # distance to kth nearest neighbor in meters
        distance = distances[-1] * (pixel_to_cm / 100)
    
        density[y, x] = (k - 1) / (math.pi * distance * distance)

    return density