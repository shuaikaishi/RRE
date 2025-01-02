import matplotlib.pyplot as plt
import cv2
import numpy as np
import sys

filename=sys.argv[1]

src = cv2.imread(filename)
src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
print(src.shape)
dsize = [512, 512]
center = [256, 256]
maxRadius = 256
base_flags =   cv2.INTER_LINEAR+cv2.WARP_FILL_OUTLIERS

radials = ["cv2.WARP_POLAR_LINEAR", 
           "cv2.WARP_POLAR_LOG", 
           "cv2.WARP_POLAR_EXP", 
           "cv2.WARP_POLAR_SQRT", 
           "cv2.WARP_POLAR_SQUARE"]

for i in range(5):
    flags = base_flags+eval(radials[i])
    radial = radials[i].split('_')[-1]

    dst = cv2.warpPolar(src, dsize, center, maxRadius, flags)

    flags = base_flags+eval(radials[i]) + cv2.WARP_INVERSE_MAP 
    rec = cv2.warpPolar(dst, dsize, center, maxRadius, flags)
 
 
    plt.subplot(3, 5, i+1)
    plt.imshow(src)
    plt.title('ori ' + radial)
    plt.axis('off')

    plt.subplot(3, 5, i+1+5)
    plt.imshow(dst)
    plt.title('polar ' + radial)
    plt.axis('off')

    plt.subplot(3,5, i+1+10)
    plt.imshow(rec)
    plt.title('rec ' + radial)
    plt.axis('off')
plt.tight_layout()
#plt.show()
plt.savefig(filename.split('.')[0]+'_res.png',dpi=600)