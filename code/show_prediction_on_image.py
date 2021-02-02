import argparse
import h5py 
import cv2 
import numpy as np

# -------------------------------------------------------------------------------------------------

# args
parser = argparse.ArgumentParser()
parser.add_argument('--positions_file', action = 'store', type = str, required = True)
parser.add_argument('--input_image_file', action = 'store', type = str, required = True)
parser.add_argument('--output_image_file', action = 'store', type = str, required = True)
parser.add_argument('--circle_radius', action = 'store', type = int, required = True)
args = parser.parse_args()

# load positions
h5_file = h5py.File(args.positions_file, 'r')
predicted_positions = np.asarray(h5_file['positions'])
   
# reading the input image
image = cv2.imread(args.input_image_file)

# red color in BGR
color = (0, 0, 255)
  
# line thickness of -1 px
thickness = -1
  
# using cv2.circle() method
# draw a circle of red color of thickness -1 px in every predicted position
for idx in range(0, len(predicted_positions)):

  print('drawing circle number {}.'.format(idx))

  coordinate = (predicted_positions[idx][0], predicted_positions[idx][1])
  image = cv2.circle(image, coordinate, args.circle_radius, color, thickness)

# using cv2.imwrite() method 
# saving the image 
cv2.imwrite(args.output_image_file, image)

print('image successfuly generated.')