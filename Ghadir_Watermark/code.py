import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

input_folder = 'Pictures/'
output_folder = 'Ghadir_WM/'
watermark_path = 'Watermark_Small.png'

os.makedirs(output_folder,exist_ok=True)

hight = 1000
width = 1500

for filename in os.listdir(input_folder) :
    file_path = os.path.join(input_folder, filename)

    img=cv2.imread(file_path)
    if img is None :
        print('Error Read Image')
        break


    wm = cv2.imread(watermark_path,cv2.IMREAD_UNCHANGED)
    if wm is None :
        print('Error Read watermark')
        break

    img_resized = cv2.resize(img,(width,hight),interpolation=cv2.INTER_CUBIC)
    img_bg = cv2.cvtColor(img_resized,cv2.COLOR_BGR2BGRA)

    watermark = np.zeros_like(img_bg,dtype=np.uint8)
    wm_h = wm.shape[0]
    wm_w = wm.shape[1]

    watermark[-wm_h-20:-20,-wm_w-50:-50,:] = wm

    watermark_bgr = watermark[:,:,0:3]
    watermark_alpha = watermark[:,:,3] / 255.0

    watermark_alpha = watermark_alpha.reshape(1000,1500,1)

    img_bg_bgr = img_bg[:,:,0:3]
    img_bg_float = img_bg_bgr.astype(np.float32) / 255.0
    watermark_bgr_float = watermark_bgr.astype(np.float32) / 255.0

    blended_rgb = img_bg_float * (1 - watermark_alpha) + watermark_bgr_float * watermark_alpha
    blended_rgb = (blended_rgb * 255).astype(np.uint8)

    output_path = os.path.join(output_folder,filename)
    print(f'{filename} : OK')
    cv2.imwrite(output_path,blended_rgb)