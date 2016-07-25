
import numpy as np
import cv2
import colorization as cz
import os
import sys

# read input parameters below
if (len(sys.argv) < 2):
    print("Need to specify a folder\n")
    exit()
image_name = sys.argv[1]
iteration = 1
threshold = 0.01
wd = 1
if (len(sys.argv) > 2):
    iteration = int(sys.argv[2])
if (len(sys.argv) > 3):
    threshold = float(sys.argv[3])
if (len(sys.argv) > 4):
    wd = int(sys.argv[4])

# set up internal parameters, don't change
display_scale = 0.7
kept_threshold  = 0.1
input_suffix = ".bmp";
marked_suffix = "_marked.bmp"
mask_suffix = "_mask.bmp"
marked_intensity_suffix = "_marked_intensity.bmp"
kept_mask_image_suffix = "_kept_mask.bmp"
kept_mask_binary_image_suffix = "_kept_mask_binary.bmp"
gray_suffix = "_gray.bmp"
out_suffix = "_res.bmp"

input_image_name = os.path.join(image_name, "image"+input_suffix)
gray_image_name = os.path.join(image_name, "image"+gray_suffix)
marked_image_name = os.path.join(image_name, "image"+marked_suffix)
kept_mask_image_name = os.path.join(image_name, "image"+kept_mask_image_suffix)
kept_mask_binary_image_name = os.path.join(image_name, "image"+kept_mask_binary_image_suffix)

parameter_suffix = "_iterations_" + str(iteration) + "_threshold_" + str(threshold) + "_wd_" + str(wd)
if os.path.isfile(kept_mask_image_name):
    parameter_suffix += "_with_kept_mask"

marked_intensity_image_name = os.path.join(image_name, "image"+parameter_suffix+marked_intensity_suffix)
mask_image_name = os.path.join(image_name, "image"+parameter_suffix+mask_suffix)
out_image_name = os.path.join(image_name, "image"+parameter_suffix+out_suffix)

# print running parameters
print "iterations: ", iteration
print "threshold: ", threshold
print "neighbor wdith: ", wd
print input_image_name
print gray_image_name
print marked_image_name
print marked_intensity_image_name
print mask_image_name
print kept_mask_image_name
print kept_mask_binary_image_name
print out_image_name

# read input and marked image
input_image_uint8 = cv2.imread(input_image_name, cv2.IMREAD_COLOR)
marked_image_uint8 = cv2.imread(marked_image_name, cv2.IMREAD_COLOR)
input_image = input_image_uint8.astype(np.float)/255
marked_image = marked_image_uint8.astype(np.float)/255
print "image size: ", input_image.shape[0], "x", input_image.shape[1]
cz.show_img(marked_image, scale = display_scale)

# read kept mask image if there is one
if os.path.isfile(kept_mask_image_name):
    kept_mask_image_uint8 = cv2.imread(kept_mask_image_name, cv2.IMREAD_COLOR)
    kept_mask_image = kept_mask_image_uint8.astype(np.float)/255
    print "running with kept mask"
    cz.show_img(kept_mask_image, scale = display_scale)
else:
    kept_mask_image = None
    print "running without kept mask"

# get gray image and the mask generated
res = cz.get_gray_mask(input_image, marked_image, threshold = threshold, \
     kept_mask_image = kept_mask_image, kept_threshold = kept_threshold )
gray_image = res["gray_image"]
mask_image = res["mask_image"]
marked_intensity_image = res["marked_intensity"]
kept_mask_binary_image = res["kept_mask_binary_image"]

# print and save gray image and masks
cz.show_img(gray_image, scale = display_scale)
#if kept_mask_binary_image is not None:
#    cz.show_img(kept_mask_binary_image, scale = display_scale)
cz.show_img(mask_image, scale = display_scale)
cz.show_img(marked_intensity_image, scale = display_scale)

cv2.imwrite(gray_image_name, gray_image*255)
cv2.imwrite(mask_image_name, mask_image*255)
cv2.imwrite(marked_intensity_image_name, marked_intensity_image*255)
if kept_mask_binary_image is not None:
    cv2.imwrite(marked_intensity_image_name, kept_mask_binary_image*255)

# running coloring algorithm
marked_image_yuv = res["marked_image_yuv"]
mark = res["mark"]
res = cz.colorize(marked_image_yuv, mark, iteration = iteration, wd = wd)
color_image = res["color_image"]
color_image_yuv = res["color_image_yuv"]

# print and save final result
cz.show_img(color_image, scale = display_scale)
cv2.imwrite(out_image_name, color_image*255)

print "done"