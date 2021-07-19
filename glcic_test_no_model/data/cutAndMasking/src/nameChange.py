import os

# 주어진 디렉토리에 있는 항목들의 이름을 담고 있는 리스트를 반환합니다.
# 리스트는 임의의 순서대로 나열됩니다.
file_path = 'C:\\Users\\dw\\github_repository\\Facial-mask-overlay-with-OpenCV-Dlib\\testing\\photo'
parentDirName = os.listdir(file_path)

i = 1
for subDirName in parentDirName:
    subdir_path='C:\\Users\\dw\\github_repository\\Facial-mask-overlay-with-OpenCV-Dlib\\testing\\photo\\'+str(subDirName)
    file_names = os.listdir(subdir_path)
    print(file_names)
    for fileName in file_names:
        src = os.path.join(subdir_path,str(fileName))
        dst = str(fileName[0:-3])+str(i) + '.jpg'
        dst = os.path.join(subdir_path, dst)
        os.rename(src, dst)
        i += 1