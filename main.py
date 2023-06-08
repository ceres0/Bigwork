import cv2
from align import align

# load
ref_img = cv2.imread('ref.jpg')
align_img = cv2.imread('align.jpg')

align(ref_img, align_img)
