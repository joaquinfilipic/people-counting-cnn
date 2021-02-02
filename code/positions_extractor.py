import json
import os
import glob
import h5py
import argparse

# -------------------------------------------------------------------------------------------------

# args
parser = argparse.ArgumentParser()
parser.add_argument('--particles_dir', action = 'store', type = str, required = True)
parser.add_argument('--positions_dir', action = 'store', type = str, required = True)
args = parser.parse_args()

# print configuration
print('configuration: {}.'.format(args))

particles_directory = args.particles_dir
positions_directory = args.positions_dir

# find all particle files in particles directory
particles_file_paths = []
for particles_file_path in glob.glob(os.path.join(particles_directory, '*.json')):
    particles_file_paths.append(particles_file_path)

current_file = 0
remaining = len(particles_file_paths)

# iterate over every particle file
for particles_file_path in particles_file_paths:
    
    # initialize array for positions
    positions = []

    print('Processed files: {}. Remaining: {}'.format(current_file, remaining))
    print('Extracting positions from file: {}.'.format(particles_file_path))

    # load particles array
    with open(particles_file_path) as json_file:
        particles = json.load(json_file)

    # add all the particles positions
    for particle in particles:
        positions.append((round(particle['x']), round(particle['y'])))

    # save positions for current simulation output file
    positions_file_path = particles_file_path.replace(particles_directory, positions_directory).replace('particles', 'positions').replace('.json', '.h5')
    with h5py.File(positions_file_path, 'w') as hf:
        hf['positions'] = positions

    current_file = current_file + 1
    remaining = remaining - 1

print('Extraction finished.')