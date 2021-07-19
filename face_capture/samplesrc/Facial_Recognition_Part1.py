import cv2
import numpy as np

# 정면 얼굴 인식
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


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
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# 카메라 실행
cap = cv2.VideoCapture(0)
count = 1

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1

        # 이미지 200x200으로 조정
        face = cv2.resize(face_extractor(frame),(200,200))

        # 흑백 -> X
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # .jpg로 캡처한 파일 저장
        file_name_path = 'faces/user'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        # 얼굴과 현재 저장개수 표시
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
        #if count%50==0:
        print(f'count : {count}')
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1)==13 or count==1000:
        break

cap.release()
cv2.destroyAllWindows()
print('Colleting Samples Complete!!!')
