# -------------------------------------------------------------------------------------------------
# This scripts blures and applies noise to all images found in the given directories.
# -------------------------------------------------------------------------------------------------

import argparse
import numpy as np
import glob
import os
import cv2
import random

# -------------------------------------------------------------------------------------------------
# Applies noise to a single pixel.
#     pixel_value: original pixel value to modify.
#     noise_eta:   amplitude of the random uniform intensity modifier.
# -------------------------------------------------------------------------------------------------
def add_noise_to_pixel(pixel_value, noise_eta):
    
    intensity_modifier = random.uniform(1.0 - noise_eta, 1.0 + noise_eta)
    pixel_value = int(pixel_value * intensity_modifier)

    if pixel_value > 255:
        pixel_value = 255
    
    return pixel_value

# -------------------------------------------------------------------------------------------------
# Adds noise to an image.
#     image:         matrix with the image.
#     noise_eta:     amplitude of the random uniform intensity modifier.
#     is_gray_image: boolean that indicates if the image is gray.
# -------------------------------------------------------------------------------------------------
def add_noise(image, noise_eta, is_gray_image):

    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):

            if is_gray_image:
                new_pixel_value = add_noise_to_pixel(image[y][x][0], noise_eta)
                image[y][x][0] = new_pixel_value
                image[y][x][1] = new_pixel_value
                image[y][x][2] = new_pixel_value
            else:
                image[y][x][0] = add_noise_to_pixel(image[y][x][0], noise_eta)
                image[y][x][1] = add_noise_to_pixel(image[y][x][1], noise_eta)
                image[y][x][2] = add_noise_to_pixel(image[y][x][2], noise_eta)
    
    return image

# -------------------------------------------------------------------------------------------------
# Applies blurring to a single image.
#     image:       matrix with the image.
#     kernel_size: size of the gaussian kernel to use in the blurring process.
#     sigma:       sigma used to initialize the gaussian kernel.
# -------------------------------------------------------------------------------------------------
def blur_image(image, kernel_size, sigma):

    # use gaussian kernel to blur image
    blurred = cv2.GaussianBlur(image, (kernel_size,kernel_size), sigmaX = sigma, sigmaY = sigma, borderType = cv2.BORDER_DEFAULT)

    return blurred

# -------------------------------------------------------------------------------------------------
# Process a single image, applying blurring, noise, and saving it in its corresponding directory.
#     image_path:     path to the image to process.
#     processed_path: path in which to save the processed image.
#     kernel_size:    size of the gaussian kernel to use in the blurring process.
#     sigma:          sigma used to initialize the gaussian kernel.
#     noise_eta:      mplitude of the random uniform intensity modifier.
#     is_gray_image:  boolean that indicates if the image should be saved in grayscale.
# -------------------------------------------------------------------------------------------------
def process_image(image_path, processed_path, kernel_size, sigma, noise_eta, is_gray_image):

    # open original image
    original_image = cv2.imread(image_path)

    # apply blurring
    blurred_image = blur_image(original_image, kernel_size, sigma)

    # add noise
    processed_image = add_noise(blurred_image, noise_eta, is_gray_image)

    # save processed image
    cv2.imwrite(processed_path, processed_image)

# -------------------------------------------------------------------------------------------------

# args
parser = argparse.ArgumentParser()
parser.add_argument('--kernel_size', action = 'store', type = int, required = True)
parser.add_argument('--sigma', action = 'store', type = int, required = True)
parser.add_argument('--noise_eta', action = 'store', type = float, required = True)
parser.add_argument('--raw_images_dir', action = 'store', type = str, required = True)
parser.add_argument('--processed_images_dir', action = 'store', type = str, required = True)
args = parser.parse_args()

# print configuration
print('configuration: {}.'.format(args))

raw_rgb_images_dir = args.raw_images_dir + '/rgb'
raw_ir_images_dir = args.raw_images_dir + '/ir'

# process all rgb images
print('Processing rgb raw images.')
rgb_images_paths = []
for image_path in glob.glob(os.path.join(raw_rgb_images_dir, '*.jpg')):
    rgb_images_paths.append(image_path)

processed_images = 0
remaining = len(rgb_images_paths)

for image_path in rgb_images_paths:

    print('Processed images: {}. Remaining: {}.'.format(processed_images, remaining))
    print('Processing image: {}.'.format(image_path))
    processed_path = image_path.replace(args.raw_images_dir, args.processed_images_dir)
    process_image(image_path, processed_path, args.kernel_size, args.sigma, args.noise_eta, is_gray_image = False)

    processed_images = processed_images + 1
    remaining = remaining - 1

# process all ir images
print('Processing ir raw images.')
ir_images_paths = []
for image_path in glob.glob(os.path.join(raw_ir_images_dir, '*.jpg')):
    ir_images_paths.append(image_path)

processed_images = 0
remaining = len(ir_images_paths)

for image_path in ir_images_paths:

    print('Processed images: {}. Remaining: {}.'.format(processed_images, remaining))
    print('Processing image: {}.'.format(image_path))
    processed_path = image_path.replace(args.raw_images_dir, args.processed_images_dir)
    process_image(image_path, processed_path, args.kernel_size, args.sigma, args.noise_eta, is_gray_image = True)

    processed_images = processed_images + 1
    remaining = remaining - 1

print('All images processed successfully.')