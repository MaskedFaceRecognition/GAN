import cv2
import numpy as np
import os
#import sys

# 정면 얼굴 인식
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#파라미터

imageSize=(128,128)
florderName=f'faces_size{imageSize[0]}'
count = 1
# imageSize : 캡쳐할 크기

def face_extractor(img):
    # 흑백 처리
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # 얼굴 찾기
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    
    # 찾은 얼굴 X -> None
    if faces is():
        return None
    
    # 찾은 얼굴 O -> Cropped Face(잘린 얼굴)
    for(x,y,w,h) in faces:
        # cropped_face = img[y-20:y+h+20, x-20:x+w+20]
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# 카메라 실행
cap = cv2.VideoCapture(0)

def makeDir(inputPath):
    if not os.path.exists(inputPath): os.makedirs(inputPath)


def capture_location(count):
    dir_name='faces2'
    makeDir(f'{dir_name}')
    return f'{dir_name}/user'+str(count)+'.jpg'


while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1

        # 이미지 조정
        face = cv2.resize(face_extractor(frame),imageSize)

        # .jpg로 캡처한 파일 저장
        file_name_path = capture_location(count)
        cv2.imwrite(file_name_path,face)

        # 얼굴과 현재 저장개수 표시
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
        if count%50==0:
            print(f'count : {count}')
    else:
        print("Face not Found")
        pass
    # enterkey를 치면 종료 or 1000장
    if cv2.waitKey(1)==13 or count==1000:
        break

cap.release()
cv2.destroyAllWindows()
print(f'Colleting Samples Complete!!!')
