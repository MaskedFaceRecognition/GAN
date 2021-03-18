# 이미지를 가져와 넘파이 파일로 변환한다.
# 
import os
import numpy as np
from PIL import Image

from PIL import Image
 
img = Image.open("user2.jpg")


np_arr = np.asarray(img)
# 넘파이 어레이로
img1 = Image.fromarray(np_arr)
# 넘파이 파일로 저장.
inputPath='faces2_array'
if not os.path.exists(inputPath): os.makedirs(inputPath)
np.save('./faces2_array/user2_array',img1)
