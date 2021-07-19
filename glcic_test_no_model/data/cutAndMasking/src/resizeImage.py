# Necessary imports
# train test 분류되어 있음
# photo 폴더 안에 shape_predictor_68_face_landmarks.dat 파일이 존재해야 한다.
# photo 폴더에 사람명 폴더가 있고 그 안에 있는 사진을 얼굴 부분만 resize한다.
# masking 변수가 True이면 마스크 씌워서 resize 한다.
import cv2
import dlib
import numpy as np
import os
import imutils
import sys

rootDirectoryName="C:\\Users\\dw\\github_repository\\senier_project_mask_to_nonMask\\cutAndMasking"

color_blue = (254,207,110)
color_cyan = (255,200,0)
color_white = (255, 255, 255) # white
color_black = (0,0,0)

choice1 = color_white # black color
choice2 = 1 # fmask_my
testNum= 6
masking=False

resizeWidth=28
resizeHeight=28

def faceProcessing(face,predictor,gray,img,masking):
    # 마스크 씌우는 문장
    if masking==True:
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

    # resize 문장
    img = img[face.top()-20:face.bottom()+10,face.left()-20:face.right()+20]
    img = imutils.resize(img, width = resizeWidth, height=resizeHeight)
    return img


def dataProcessing(subDirName,fileName,index):
    global rootDirectoryName
    # 파일 처리, 이미지 경로를 설정해준다.
    try:
        if fileName[-3:]=="jpg":
            #Initialize color [color_type] = (Blue, Green, Red)
            inputImagePath = rootDirectoryName+"\\photo\\"+str(subDirName)+"\\"+str(fileName)

            # Loading the image and resizing, converting it to grayscale
            img= cv2.imread(inputImagePath)
            gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Initialize dlib's face detector
            detector = dlib.get_frontal_face_detector()
            faces = detector(gray, 1)

            p = "shape_predictor_68_face_landmarks.dat"
            predictor = dlib.shape_predictor(p)

            # 얼굴 랜드마킹 한다. 중요 함수는 dlib.shape_predictor(p)
            # 얼굴을 확인한 다음 랜드마킹할 곳을 확인
            for face in faces:
                img3=faceProcessing(face,predictor,gray,img,masking)
                
                #Save the output file to testing
                # 여기서 마스킹 할때 맨 뒤에 .jpg를 붙여줘야 한다
                if masking==True:
                    if index<testNum:
                        outputNameofImage = rootDirectoryName+"\\maskedOutput\\test\\"+subDirName+"\\"+"masked_"+subDirName+"_"+str(index)+".jpg"
                    else:
                        outputNameofImage = rootDirectoryName+"\\maskedOutput\\train\\"+subDirName+"\\"+"masked_"+subDirName+"_"+str(index)+".jpg"
                elif masking==False:
                    if index<testNum:
                        outputNameofImage=rootDirectoryName+"\\nonMaskedOutput\\test\\"+subDirName+"\\"+subDirName+"_"+str(index)+".jpg"
                    else:
                        outputNameofImage=rootDirectoryName+"\\nonMaskedOutput\\train\\"+subDirName+"\\"+subDirName+"_"+str(index)+".jpg"
                cv2.imwrite(outputNameofImage, img3)
    except:
        print("excepted"+fileName)
def makeDir(inputPath):
    if not os.path.exists(inputPath): os.makedirs(inputPath)


def mkAllDir(masking):
    makeDir("../maskedOutput")
    makeDir("../nonMaskedOutput")
    if masking==True:
        makeDir("../maskedOutput/test")
        makeDir("../maskedOutput/train")
        makeDir("../maskedOutput/test/"+str(subDirName))
        makeDir("../maskedOutput/train/"+str(subDirName))
    elif masking==False:
        makeDir("../nonMaskedOutput/test")
        makeDir("../nonMaskedOutput/train")
        makeDir("../nonMaskedOutput/test/"+str(subDirName))
        makeDir("../nonMaskedOutput/train/"+str(subDirName))


def setPath(imputPath):
    os.chdir(imputPath)

if __name__ == "__main__":
    # 얼굴 마스킹
    file_path2 = rootDirectoryName+"\\photo"
    parentDirName2 = os.listdir(file_path2) # 사람 이름 폴더
    printProgress=len(parentDirName2)
    
    print("테스트 데이터 개수 : "+str(testNum-1))
    print("마스킹 유무 : "+str(masking))
    print(" 마스킹 방법 : "+str(choice1))
    for index,subDirName in enumerate(parentDirName2):
        print(subDirName)
        print("총 "+str(printProgress)+"폴더 중" +str(index+1)+"번째 폴더 진행중")
        count=0
        if subDirName[-4:]!=".dat":
            mkAllDir(masking)
            os.chdir(file_path2)
            file_names = os.listdir(subDirName)
            #setPath(file_path2)
            for fileName in file_names:
                count+=1
                #print(fileName)
                dataProcessing(subDirName,fileName,count)
