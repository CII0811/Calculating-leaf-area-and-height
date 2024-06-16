import cv2
import numpy as np
import os
import pandas as pd

#計算高度的副程式
def heigh(gray_img, new_height, new_width, threshold_value=90):
    # 在圖像下半1/3添加黑色遮罩
    mask_height = int(new_height * 0.32)
    black_mask = np.zeros((mask_height, new_width, 3), dtype=np.uint8)  
    # 黑色遮罩轉換為灰度圖像
    black_mask_gray = cv2.cvtColor(black_mask, cv2.COLOR_BGR2GRAY)
    # 將黑色灰度遮罩應用到圖像的下半部分
    gray_img[-mask_height:] = black_mask_gray 
    # 灰度圖像轉換為二值圖像(去掉白色背景部分，設定白色背景的閥值範圍)
    _, mask = cv2.threshold(gray_img, threshold_value, 255, 
                            cv2.THRESH_BINARY)
    #反轉二值圖像
    mask_inv = cv2.bitwise_not(mask)
    #下半部分設置為黑色
    mask_inv[-mask_height:] = 0
    # 查找輪廓
    contours, _ = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, 
                                   cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大的矩形
    max_rect = None
    max_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > max_area:
            max_area = area
            max_rect = (x, y, w, h)
    # 在圖像上畫出最大的矩形
    if max_rect is not None:
        x, y, w, h = max_rect
        cv2.rectangle(gray_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        height = h // 21.5
        # 結果輸出，外接矩形的高度
        print(f"{file_name} 高度: {height} cm")
        return height
    return None

#計算葉面積的副程式，距離變換和分水領算法分割圖像的前景和背景
def leaf_area(gray_img):
    #生成二值化圖像
    ret,thresh_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV )
    kernel = np.ones((3,3), np.uint8)
    #去除雜訊
    open_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=2)
    #距離變換圖像
    dst_img = cv2.distanceTransform(open_img, cv2.DIST_L2, maskSize=5)
    #得到前景區域
    ret, sure_fg = cv2.threshold(dst_img, 0.7 * dst_img.max(), 255, cv2.THRESH_BINARY)
    #得到擴大的背景區域
    sure_bg = cv2.dilate(open_img, kernel, iterations=3)
    #擴大背景區域與前景區域的差異
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    #標記前景
    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0
    #使用分水嶺算法對圖像進行分割
    watershed_img = cv2.watershed(small_img, markers)
    #標記邊界區域
    small_img[watershed_img == -1] = [0, 0, 255]
    #創建黑色遮罩
    mask = np.zeros(gray_img.shape, dtype=np.uint8)
    #分割結果中標記值大於 1 的區域設置為白色代表葉片
    mask[markers > 1] = 255
    #應用到圖像中
    masked_img = cv2.bitwise_and(small_img, small_img, mask=mask)
    #轉換為灰度圖像
    masked_gray_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    #生成二值圖像
    ret, masked_thresh = cv2.threshold(masked_gray_img, 0, 255, 
                                       cv2.THRESH_BINARY )
    #計算白色區域像素
    area = cv2.countNonZero(masked_thresh)
    real_area = area // 261
    print(f"{file_name} 面積: {real_area} mc²")
    return real_area

# 指定資料夹
folder_path = "leaf"

# 讀取資料夾中的所有檔名
file_names = os.listdir(folder_path)

# 依照第8、9、10個(編號)字排序，然後照前2個字（天數）排序
file_names.sort(key=lambda x: (x[7:10],x[:2]))

results = []

for file_name in file_names:
    # 讀取圖像
    img_path = os.path.join(folder_path, file_name)
    img = cv2.imread(img_path)

    # 取得圖像的原始尺寸
    original_height, original_width, original_channels = img.shape

    # 缩小圖像(*0.2)
    new_width = int(original_width * 0.2)
    new_height = int(original_height * 0.2)
    small_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    #轉換為灰度
    gray_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)

    #檔名的天數、重量、編號也要寫進excel中
    day = int(file_name[:2])
    group = file_name[8:10]
    weight = file_name[3:6]

    #依照規定分別計算正面照(0)的面積和側面照(1)的高度，並將結果添加到results列表中
    if file_name[7] == '0':
        area=leaf_area(gray_img)
        if area is not None:
            results.append({'Plant ID': group,'day': day,  'type': 'area', 'value': area,'weight': weight})
    elif file_name[7] == '1':
        height=heigh(gray_img, new_height, new_width)
        if height is not None:
            results.append({'Plant ID': group, 'day': day, 'type': 'height', 'value': height,'weight': weight})

#將結果轉換為 Pandas DataFrame
df = pd.DataFrame(results)

# 調整 DataFrame 的結構，使同一張照片的面積和高度在同一橫排
df = df.pivot_table(index=['Plant ID','day', 'weight'], columns='type', values='value').reset_index()
#保存到 Excel 文件中
df.to_excel('leaf_results.xlsx', index=False)