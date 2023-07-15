import cv2
import numpy as np

im = cv2.imread("./train/0.png")
h,w,chn = im.shape
seed = (w/2,h/2)

mask = np.zeros((h+2,w+2),np.uint8)

floodflags = 4
floodflags |= cv2.FLOODFILL_MASK_ONLY
floodflags |= (255 << 8)

num,im,mask,rect = cv2.floodFill(im, mask, seed, (255,0,0), (10,)*3, (10,)*3, floodflags)

cv2.imwrite("seagull_flood.png", mask)
