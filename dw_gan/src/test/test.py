
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras

plt.style.use('seaborn')
sns.set(font_scale = 2.5)

import warnings; warnings.filterwarnings('ignore')

%matplotlib inline

import os

print(tf.__version__)

from google.colab import drive
drive.mount('/content/gdrive/')

'''
!cd /content/gdrive/MyDrive/GraduationProject 
# MyDrive(내 드라이브)안에 바로가기를 만들면 된다. => "!"추가하여 바로가기 생성하지 않아도됨.
'''

from subprocess import check_output
print(check_output(['ls']).decode('utf8'))

os.chdir('/content/gdrive/MyDrive/GraduationProject/data/Train_Test/forTest/train/')
print(check_output(['ls']).decode('utf8')


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import cv2
import glob
from google.colab.patches import cv2_imshow

train_images = []
train_target_str = []
train_target = [] # 0부터 저장됨
idx = 0
data_folder = '/content/gdrive/MyDrive/GraduationProject/data/Train_Test/forTest/train'
folder_list = os.listdir(data_folder)
for folder in folder_list:
  print(folder)
  train_target_str.append(folder)
  files = glob.glob(os.path.join(folder, '*.jpg')) # 현재 경로의 것을 가져온다.
  for filename in files:
    temp = cv2.imread(filename, 0)
    train_images.append(temp)
    train_target.append(idx)
  idx += 1

img = train_images[0]
print(cv2_imshow(img))
print(img)


print(train_target)


os.chdir('/content/gdrive/MyDrive/GraduationProject/data/Train_Test/forTest/test/')
print(check_output(['ls']).decode('utf8'))

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import cv2
import glob
from google.colab.patches import cv2_imshow

test_images = []
test_target_str = []
test_target = [] # 0부터 저장됨
idx = 0
data_folder = '/content/gdrive/MyDrive/GraduationProject/data/Train_Test/forTest/test'
folder_list = os.listdir(data_folder)
for folder in folder_list:
  print(folder)
  test_target_str.append(folder)
  files = glob.glob(os.path.join(folder, '*.jpg')) # 현재 경로의 것을 가져온다.
  for filename in files:
    temp = cv2.imread(filename, 0)
    test_images.append(temp)
    test_target.append(idx)
  idx += 1
  
img = test_images[0]
print(cv2_imshow(img))
print(img)