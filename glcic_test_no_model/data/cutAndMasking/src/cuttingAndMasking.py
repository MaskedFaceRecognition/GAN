# Necessary imports
import cv2
import dlib
import numpy as np
import os
import imutils
import sys

rootDirectoryName="C:\\Users\\dw\\github_repository\\senier_project\\senier_project_mask_to_nonMask\\cutAndMasking"

def faceExtract(subDirName,fileName):
    global rootDirectoryName
    ## set directories
    nowPath=os.getcwd()
    os.chdir(nowPath)
    imagename=str(fileName)
    #path = os.getcwd()+"\\"+"photo\\"+subDirName+"\\"+imagename
    path = "..\\"+"photo\\"+subDirName+"\\"+imagename
    # 입력 파일 지정하기
    image_file = "..\\"+"photo\\"+subDirName+"\\"+imagename
    print(image_file)
    # 캐스케이드 파일의 경로 지정하기 --- (※1)
    cascade_file = "haarcascade_frontalface_alt.xml"
    cascade= cv2.CascadeClassifier( os.path.join(cv2.data.haarcascades, cascade_file) )
    # 이미지 읽어 들이기 --- (※2)
    image = cv2.imread(image_file)
    # 그레이스케일로 변환하기

    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 얼굴 인식 실행하기
    face_list = cascade.detectMultiScale(image_gs,
        scaleFactor=1.1,
        minNeighbors=1,
        minSize=(150,150))
    if len(face_list) > 0:
        # 인식한 부분 표시하기 --- (※4)
        #print(face_list)
        color = (0, 0, 255)
        for face in face_list:
            x,y,w,h = face
            #cv2.rectangle(image, (x,y), (x+w, y+h), color, thickness=8)
            cropImage=image[y:y+h,x:x+w]
        # 파일로 출력하기 --- (※5)
        outputNameofImage = rootDirectoryName+"\\cuttedOutput\\"+subDirName+"\\"+"cuted_"+fileName
        cv2.imwrite(outputNameofImage, cropImage)
    else:
        print("no face")

def face_Masking(subDirName,fileName):
    global rootDirectoryName
    try:
        if fileName[-3:]=="jpg":
            inputImagePath = rootDirectoryName+"\\cuttedOutput\\"+str(subDirName)+"\\"+str(fileName)
            #Initialize color [color_type] = (Blue, Green, Red)
            color_blue = (254,207,110)
            color_cyan = (255,200,0)
            color_black = (255, 255, 255) # white

            choice1 = 2 # black color
            choice2 = 1 # fmask_my

            # Loading the image and resizing, converting it to grayscale
            img= cv2.imread(inputImagePath)
            img = imutils.resize(img, width = 500)
            gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Initialize dlib's face detector
            detector = dlib.get_frontal_face_detector()
            faces = detector(gray, 1)

            p = "shape_predictor_68_face_landmarks.dat"
            predictor = dlib.shape_predictor(p)

            # 얼굴 랜드마킹 한다. 중요 함수는 dlib.shape_predictor(p)
            # 얼굴을 확인한 다음 랜드마킹할 곳을 확인
            for face in faces:
                landmarks = predictor(gray, face)
                points = []
                for i in range(1, 16):
                    point = [landmarks.part(i).x, landmarks.part(i).y]
                    points.append(point)
                # print(points)

                # Coordinates for the additional point for wide, medium coverage mask - in sequence
                mask_c = [((landmarks.part(29).x), (landmarks.part(29).y))] # 굳이 추가할 필요는 없다 
                mask_my= [((landmarks.part(15).x), (landmarks.part(15).y))]

                fmask_basic= points
                fmask_c = points + mask_c
                fmask_my= points + mask_my

                # 마스크 형식 설정 fmask_basic으로 해도 무방
                fmask_basic = np.array(fmask_basic, dtype=np.int32)
                fmask_c = np.array(fmask_c, dtype=np.int32)
                fmask_my = np.array(fmask_my, dtype=np.int32)
                mask_type = {0: fmask_basic, 1: fmask_c, 2: fmask_my}

                img2 = cv2.polylines(img, [mask_type[choice2]], True, choice1, thickness=2, lineType=cv2.LINE_8)
                img3 = cv2.fillPoly(img2, [mask_type[choice2]], choice1, lineType=cv2.LINE_AA)

            #cv2.imshow("image with mask", img3)
            
            #Save the output file to testing
            outputNameofImage = rootDirectoryName+"\\maskedOutput\\"+subDirName+"\\"+"masked_"+fileName
            cv2.imwrite(outputNameofImage, img3)
            #cv2.destroyAllWindows()
    except:
        print("except"+fileName)        

def makeDir(inputPath):
    if not os.path.exists(inputPath): os.makedirs(inputPath)

def setPath(imputPath):
    os.chdir(imputPath)

if __name__ == "__main__":
    file_path = rootDirectoryName+"\\photo"
    parentDirName = os.listdir(file_path)
    # 얼굴 추출
    '''
    for subDirName in parentDirName:
        # 샘플 이미지를 출력할 폴더
        makeDir("../cuttedOutput/"+str(subDirName))
        
        subdir_path=rootDirectoryName+"\\photo\\"+str(subDirName)
        file_names = os.listdir(subdir_path)
        for _,fileName in enumerate(file_names):
            faceExtract(subDirName,fileName)
    '''
    # 얼굴 마스킹
    file_path2 = rootDirectoryName+"\\cuttedOutput"
    parentDirName2 = os.listdir(file_path2) # 사람 이름 폴더

    for subDirName in parentDirName2:
        print(subDirName)
        if subDirName[-4:]!=".dat":
            makeDir("../maskedOutput/"+str(subDirName))
            os.chdir(file_path2)
            file_names = os.listdir(subDirName)
            #setPath(file_path2)
            for fileName in file_names:
                #print(fileName)
                face_Masking(subDirName,fileName)
