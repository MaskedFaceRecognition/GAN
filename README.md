# 
20210429
glcic_test/src-upgraded/train.py파일 수정중
(object detection과 glcic 모델 합하는 작업)

/data/images의 test, train 폴더에 사진 데이터가 있고, 이를 to_npy_2.py를 통해 batch 파일로 /npy/x_test.npy와 /npy/x_train.npy 로 만들면 이파일을 train에서 가져온다.