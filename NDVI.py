#import libraries
import cv2
import numpy as np
from fastiecm import fastiecm

#choose the specific photo to be converted to NDVI heatmap
original = cv2.imread('/home/teamname/Downloads/Houston_USA-Original.jpg')

#display a photo on the screen function
def display(image, image_name):
    image = np.array(image,dtype=float)/float(255)
    shape = image.shape
    height = int(shape[0]/2)
    width = int(shape[1]/2)
    image = cv2.resize(image, (width, height))
    cv2.namedWindow(image_name)
    cv2.imshow(image_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#increase contrast function
def contrast_stretch(Original):
    in_min = np.percentile(Original, 5)
    in_max = np.percentile(Original,95)

    out_min = 0.0
    out_max = 255.0

    out = Original - in_min
    out *= ((out_min - out_max) / (in_min - in_max))
    out += in_min

    return out

#calculate NDVI values for contrasted photo function
def calc_ndvi(image):
    b, g, r = cv2.split(image)
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom==0] = 0.01
    ndvi = (b.astype(float) - r) / bottom
    return ndvi

#actual commands to open normal, contrasted, NDVI, NDVI-contrasted and fastie heatmap photos
display(original, 'Original')
contrasted = contrast_stretch(original)
display(contrasted, 'Contrasted original')
cv2.imwrite('contrasted.png', contrasted)
ndvi = calc_ndvi(contrasted)
display(ndvi, 'NDVI')
ndvi_contrasted = contrast_stretch(ndvi)
display(ndvi_contrasted, 'NDVI Contrasted')
cv2.imwrite('ndvi_contrasted.png', ndvi_contrasted)
color_mapped_prep = ndvi_contrasted.astype(np.uint8)
color_mapped_image = cv2.applyColorMap(color_mapped_prep, fastiecm)
display(color_mapped_image, 'Color mapped')
cv2.imwrite('color_mapped_image.png', color_mapped_image)