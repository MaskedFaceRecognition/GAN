import os
import numpy as np
# clear[201210318]
# to_num.py 파일에서 만든 npy파일을 겨져와 load한다. train에서 호출됨
def load(dir_='../data/npy'):
    x_train = np.load(os.path.join(dir_, 'x_train.npy'))
    x_test = np.load(os.path.join(dir_, 'x_test.npy'))
    return x_train, x_test


if __name__ == '__main__':
    x_train, x_test = load()
    print(x_train.shape)
    print(x_test.shape)

