
# to_npy_2.py를 실행시키기 바랍니다.
import glob
import os
import cv2
import numpy as np
from PIL import Image
ratio = 0.8
image_size = 128

x = []
paths = glob.glob('./images/image/*')
for path in paths:
    img = cv2.imread(path)
    img = cv2.resize(img, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x.append(img)

x = np.array(x, dtype=np.uint8)
np.random.shuffle(x)

p = int(ratio * len(x))

x_train = x[:p]
x_test = x[p:]

if not os.path.exists('./images/train'):
    os.mkdir('./images/train')

if not os.path.exists('./images/test'):
    os.mkdir('./images/test')

for i in range(len(x)):
    im=Image.fromarray(x[i])
    if p>i:
        im.save(f'./images/train/{i}.jpg')
    else:
        im.save(f'./images/test/{i}.jpg')
