# Necessary imports
# updata 20210604
import cv2
import dlib
import numpy as np
import os
import imutils
import sys

rootDirectoryName="C:\\Users\\dw\\github_repository\\senier_project\\GAN\\glcic_test\\data\\cutAndMasking"
created_image_path="C:\\Users\\dw\\github_repository\\senier_project\\GAN\\glcic_test\\data\\images"
inputImagePath = "C:\\Users\\dw\\github_repository\\senier_project\\GAN\\glcic_test\\data\\images\\test"

color_blue = (254,207,110)
color_cyan = (255,200,0)
color_white = (255, 255, 255) # white
color_black = (0,0,0)

choice1 = color_white # black color
choice2 = 1 # fmask_my


def faceProcessing(face,predictor,gray,img):
    landmarks = predictor(gray, face)
    points = []
    for i in range(1, 16):
        point = [landmarks.part(i).x, landmarks.part(i).y]
        points.append(point)
    # print(points)

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
        img = cv2.polylines(img, [mask_type[choice2]], True, choice1, thickness=2, lineType=cv2.LINE_8)
        img = cv2.fillPoly(img, [mask_type[choice2]], choice1, lineType=cv2.LINE_AA)    
    
    img = img[max(face.top()-20,0):face.bottom()+10,max(0,face.left()-20):face.right()+20]
    img = imutils.resize(img, width = 128, height = 128)

    return img


def dataProcessing(fileName,index):
    global rootDirectoryName
    try:
        
        if fileName[-3:]=="jpg":
            # Loading the image and resizing, converting it to grayscale
            img= cv2.imread(inputImagePath+'\\'+fileName)
            gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Initialize dlib's face detector
            detector = dlib.get_frontal_face_detector()
            faces = detector(gray, 1)
            p_path= "C:\\Users\\dw\\github_repository\\senier_project\\GAN\\glcic_test\\data\\cutAndMasking\\src\\shape_predictor_68_face_landmarks.dat"
            p = "shape_predictor_68_face_landmarks.dat"
            predictor = dlib.shape_predictor(p_path)

            # 얼굴 랜드마킹 한다. 중요 함수는 dlib.shape_predictor(p)
            # 얼굴을 확인한 다음 랜드마킹할 곳을 확인
            for face in faces:
                img3=faceProcessing(face,predictor,gray,img)
                outputNameofImage = created_image_path+"\\test_masked\\"+str(index)+".jpg"
                cv2.imwrite(outputNameofImage, img3)
            print(f'{fileName} saved')
    except:
        print("excepted"+fileName)
def makeDir(inputPath):
    if not os.path.exists(inputPath): os.makedirs(inputPath)


def setPath(imputPath):
    os.chdir(imputPath)

if __name__ == "__main__":
    file_names = os.listdir(inputImagePath)
    count=0
    makeDir('C:\\Users\\dw\\github_repository\\senier_project\\GAN\\glcic_test\\data\\images\\test_masked')
    os.chdir(inputImagePath)

    for fileName in file_names:
        count+=1
        dataProcessing(fileName,count)
