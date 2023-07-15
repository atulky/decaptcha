from skimage import io
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops
from skimage.morphology import square, dilation
from shapely import geometry
from shapely.ops import unary_union

import matplotlib.pyplot as plt

def preprocessCaptcha(img):
    hsv = rgb2hsv(img[:,:,0:3])
    value_img = hsv[:, :, 2]
    mask = value_img < 0.7
    mask = dilation(mask, square(5))
    label_image = label(mask)
    
    coords = []
    for region in regionprops(label_image):
        if 100 < region.area  < 10000:
            minr, minc, maxr, maxc = region.bbox
            x = minc
            y = minr
            w = maxc - minc
            h = maxr - minr
            coords.append([minc,maxr,maxc,minr])

    polygons = []
    for coord in coords:
        polygon1 = geometry.box(*coord, ccw=True)
        polygons.append(polygon1)

    merged = unary_union(polygons)

    letter_image_regions = []
    for poly in merged.geoms:
        bbox = poly.envelope.exterior.bounds #minr, minc, maxr, maxc
        minr, minc, maxr, maxc = bbox
        miny, minx, maxy, maxx = minr, minc, maxr, maxc
        
        x = int(miny)
        y = int(minx)
        w = int(maxx - minx)
        h = int(maxy - miny)
        letter_image_regions.append((x, y, w, h))
    letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    
    letters = []
    for letter_bounding_box in letter_image_regions:
        # Grab the coordinates of the letter in the image
        x, y, w, h = letter_bounding_box
        # Extract the letter from the original image with a 5-pixel margin around the edge
        letter_image = mask[y - 5:y + h + 5, x - 5:x + w + 5]
        letters.append(letter_image)


    return letters


orig = io.imread("./train2/0008.png")
letters = preprocessCaptcha(orig)

fig, axs = plt.subplots(3)
fig.suptitle('Vertically stacked subplots')
axs[0].imshow(letters[0])
axs[1].imshow(letters[1])
axs[2].imshow(letters[2])
plt.show()
