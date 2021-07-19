import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Train Data & Data의 Label
Training_Data, Labels = [], []
print(onlyfiles)
# 경로 이용해서 이미지 불러오기
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype = np.uint8))
    Labels.append(i)

# Labels를 32bit 정수로 변환
Labels = np.asarray(Labels, dtype = np.int32)
print(Labels)
# Model 생성
model = cv2.face.LBPHFaceRecognizer_create()

# 학습 시작
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!")


