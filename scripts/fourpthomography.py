import cv2
import numpy as np

img = cv2.imread('/home/haresh/left0000.jpg', cv2.IMREAD_COLOR)


pts_img = np.asarray([
    [481, 276],
    [819, 276],
    [976, 468],
    [328, 471],
])

pts_dst = np.array([[0, 0],[299, 0],[299, 399],[0, 399]])

h, status = cv2.findHomography(pts_img, pts_dst)

scale = 0.45

S = np.asarray([[scale, 0, 280],[0, scale, 360], [0, 0, 1]])

h = np.matmul(S, h, np.linalg.inv(S))

print('h : ', h)

im_out = cv2.warpPerspective(img, h, (720, 640))

cv2.imshow('source',img)
cv2.imshow('target', im_out)
cv2.waitKey(0)


