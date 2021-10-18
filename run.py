import sys
import os

import cv2
import json
import numpy as np

import retinex

data_path = 'data'
save_path = 'tran_data'

img_list = os.listdir(data_path)

if len(img_list) == 0:
    print ('Data directory is empty.')
    exit()

with open('config.json', 'r') as f:
    config = json.load(f)

for img_name in img_list:
    
    if img_name == '.gitkeep':
        continue
    
    
    img_array = np.fromfile(os.path.join(data_path, img_name), np.uint8)
    print(img_array.shape)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    print(img.shape)
    # img_msrcr = retinex.MSRCR(
    #     img,
    #     config['sigma_list'],
    #     config['G'],
    #     config['b'],
    #     config['alpha'],
    #     config['beta'],
    #     config['low_clip'],
    #     config['high_clip']
    # )
   
    # img_amsrcr = retinex.automatedMSRCR(
    #     img,
    #     config['sigma_list']
    # )

    img_msrcp = retinex.MSRCP(
        img,
        config['sigma_list'],
        config['low_clip'],
        config['high_clip']        
    )    

    shape = img.shape
    #cv2.imshow('Image', img)
    # cv2.imshow('retinex', img_msrcr)
    # cv2.imshow('Automated retinex', img_amsrcr)
    # cv2.imshow('MSRCP', img_msrcp)
    # cv2.waitKey()

    # imwrite 저장시 한글 문제 해결 방법 전용 함수
    def imwrite(filename, img, params=None):
        try: 
            ext = os.path.splitext(filename)[1] 
            result, n = cv2.imencode(ext, img, params) 

            if result: 
                with open(filename, mode='w+b') as f: 
                    n.tofile(f) 
                    return True 
            else: 
                return False 
        except Exception as e: 
            print(e) 
            return False

    img_name = 'traned_' + img_name 
    #cv2.imwrite(os.path.join(save_path, img_name), img_msrcp)
    imwrite(os.path.join(save_path, img_name), img_msrcp)