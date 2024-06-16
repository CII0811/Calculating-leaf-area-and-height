import cv2
import numpy as np
from findPoints import draw

filename='30cm.jpg'
img=cv2.imread(filename)

#convert to gray,canny to find edge
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,100,200)
#draw('canny edge',edges)

#HoughlineP
lines=cv2.HoughLinesP(edges,rho=1,theta=np.pi/180,threshold=70,
                      minLineLength=10,maxLineGap=200)
for line in lines:
    x1,y1,x2,y2=line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),8)
#draw('HoughlineP',img)


#將圖像縮小至原來的 20% 大小
original_height, original_width, original_channels = img.shape
new_width = int(original_width * 0.2)
new_height = int(original_height * 0.2)
small_img = cv2.resize(img, (new_width, new_height),
                        interpolation=cv2.INTER_AREA)
height,width, channels = small_img.shape
#draw('small_img',small_img)

# 創建黑色遮罩
mask = np.zeros(small_img.shape[:2], dtype=np.uint8)

# 定義矩形的左上角和右下角
top_left = (246, 307)
bottom_right = (350, 495)

# 在遮罩中間挖洞
cv2.rectangle(mask, top_left, bottom_right, 255, -1)

# 將遮罩應用在small_img上
masked_img = cv2.bitwise_and(small_img, small_img, mask=mask)
draw('small_img_mask',masked_img)

#黑色和红色的BGR颜色
lower_black = np.array([0, 0, 0], dtype=np.uint8)
upper_black = np.array([30, 30, 30], dtype=np.uint8) 
lower_red = np.array([0, 0, 160], dtype=np.uint8)
upper_red = np.array([80, 80, 255], dtype=np.uint8) 

mask_black = cv2.inRange(masked_img, lower_black, upper_black)
mask_red = cv2.inRange(masked_img, lower_red, upper_red)

#非黑非紅變白色
final_mask = cv2.bitwise_or(mask_black, mask_red)
result_img = masked_img.copy()
result_img[final_mask != 255] = [255, 255, 255] 

#算白色像素
white_pixel_count = np.sum(np.all(result_img == [255, 255, 255], axis=-1)) 

draw('result_img', result_img)
print(f'白色像素：{white_pixel_count}')