# -*- codeing: utf-8 -*-
import sys
import os
import cv2
import dlib
import glob

input_dir = './dog'
output_dir = './other_dogfaces1'
size = 64

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#使用dlib训练的狗脸探测器作为我们的特征提取器
detector = dlib.simple_object_detector("detector.svm")

current_path = os.getcwd()
test_folder = current_path + '/dog/'

for f in glob.glob(test_folder+'*.JPG'):
    print("Processing file: {}".format(f))
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])
    # 使用detector进行狗脸检测 dets为返回的结果
    dets = detector(img2)
        #for index, face in enumerate(dets):
        #print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))
    
    #使用enumerate 函数遍历序列中的元素以及它们的下标
    #下标i即为人脸序号
    #left：人脸左边距离图片左边界的距离 ；right：人脸右边距离图片左边界的距离
    #top：人脸上边距离图片上边界的距离 ；bottom：人脸下边距离图片上边界的距离
    for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                # img[y:y+h,x:x+w]
                face = img[x1:y1,x2:y2]
                # 调整图片的尺寸
                face = cv2.resize(face, (size,size))
                cv2.imshow('image',face)
                # 保存图片
                
                cv2.imwrite(output_dir+'/'+str(x1)+str(y1)+str(x2)+str(y2)+'.jpg', face)
    
            
    key = cv2.waitKey(30) & 0xff
    if key == 27:
       sys.exit(0)


