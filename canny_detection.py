import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np

img = mpimg.imread('road.jpg')

"""
plt.figure(figsize=(10,8))
print('This image is: ',type(img), 'with dimensions:', img.shape)
plt.imshow(img)
plt.show()
"""

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

gray = grayscale(img)
#plt.figure(figsize=(10,8)) # figsize: 창 크기
#plt.imshow(gray,cmap='gray') # cmap: 푸르딩딩한 사진 안보게
#plt.show()

def gaussian_blur(img,kernel_size):
    return cv2.GaussianBlur(img, (kernel_size,kernel_size),0)

kernel_size = 5
blur_gray = gaussian_blur(gray, kernel_size)

#plt.figure(figsize=(10,8))
#plt.imshow(blur_gray, cmap = 'gray')
#plt.show()

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

low_threshold = 50
high_threshold = 200
edges = canny(blur_gray, low_threshold, high_threshold)

#plt.figure(figsize=(10,8))
#plt.imshow(edges, cmap='gray')
#plt.show()


#plt.figure(figsize=(10,8))
#plt.imshow(mask, cmap='gray')
#plt.show()

#print(img.shape) # (363, 648, 3)

def region_of_interest(img,vertices):
    
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2] # 채널 수
        ignore_mask_color = (255,)*channel_count # (255,255,255)
        #print(ignore_mask_color)
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask,vertices,ignore_mask_color) # cv2.fillPoly: 다각형 그리기
    masked_image = cv2.bitwise_and(img, mask) # mask pixel이 nonzero 인 부분만 반환 
    return masked_image

imshape =img.shape
#print(imshape)

vertices = np.array([[(0,imshape[0]-80),
                        (230,120),
                        (420,120),
                        (imshape[1],imshape[0]-80)]], dtype = np.int32) # vertice 사다리꼴 꼭지점

mask = region_of_interest(edges, vertices)

plt.figure(figsize=(10,8))
plt.imshow(mask,cmap='gray')
plt.show()