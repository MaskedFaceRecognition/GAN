import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
# 참고 https://m.blog.naver.com/PostView.nhn?blogId=stop2y&logNo=221529660467&proxyReferer=https:%2F%2Fwww.google.com%2F
#실제 이미지, 라벨 파싱
###############################################
train_list , test_list = [], []
with open('train.txt') as f:
    for line in f:
        tmp = line.strip().split()
        #[0] = jpg 파일이름
        #[1] = 과일 인덱스  ex) 바나나 = 1 사과 = 0
        train_list.append([tmp[0], tmp[1]])

with open('test.txt') as f:
    for line in f:
        tmp = line.strip().split()
        test_list.append([tmp[0], tmp[1]])
###############################################

def readimg(path):
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    #img = plt.imread(path,cmap="gray") #imread는 이미지를 다차원 Numpy 배열로 로딩한다.
    img=np.reshape(img, [-1, 10000])
    return img


def batch(train_list, batch_size):
    img, label, paths = [], [], []
    for i in range(batch_size):
        img.append(readimg(train_list[0][0]))
        label_list= [0 for _ in range(n_class)]
        label_list[int(train_list[0][1])]=int(train_list[0][1])
        label.append(label_list)

        path.append(train_list.pop(0))
    return img, label


#옵션
###############################################
n_input=100*100
n_class=3
n_noise=128
total_epoch = 10
batch_size = 1461
learning_rate = 0.0002
n_hidden = 256
###############################################

#신경만 모델 구성
###############################################
# added and change
tf.compat.v1.disable_eager_execution()
X = tf.compat.v1.placeholder(tf.float32, [None, n_input])
Y = tf.compat.v1.placeholder(tf.float32, [None, n_class])
Z = tf.compat.v1.placeholder(tf.float32, [None, n_noise])
###############################################

#생성자
def generator(noise, labels):
    with tf.compat.v1.variable_scope('generator'):
        # noise 값에 labels 정보를 추가합니다.
        inputs = tf.concat([noise, labels], 1) #noise 값에 label 추가

        # TensorFlow 에서 제공하는 유틸리티 함수를 이용해 신경망을 매우 간단하게 구성할 수 있습니다.
        # tf.keras.layers.Dense로 바꿔서 사용하라고 한다.
        hidden = tf.compat.v1.layers.dense(inputs, n_hidden,
                                 activation=tf.nn.relu)
        output = tf.compat.v1.layers.dense(hidden, n_input,
                                 activation=tf.nn.sigmoid)
    return output

#구분자
def discriminator(inputs, labels, reuse=None):
    with tf.compat.v1.variable_scope('discriminator') as scope:
        # 노이즈에서 생성한 이미지와 실제 이미지를 판별하는 모델의 변수를 동일하게 하기 위해,
        # 이전에 사용되었던 변수를 재사용하도록 합니다.
        if reuse:
            scope.reuse_variables()

        inputs = tf.concat([inputs, labels], 1)

        hidden = tf.compat.v1.layers.dense(inputs, n_hidden,
                                 activation=tf.nn.relu)
        output = tf.compat.v1.layers.dense(hidden, 1, # 판별기의 최종 결과값은 얼마나 진짜와 가깝냐를 판단하는 한 개의 스칼라값입니다.
                                 activation=None) #활성화 함수를 사용하지 않은 이유는 뒤에 나옴

    return output

#노이즈 생성
###############################################
def get_noise(batch_size, n_noise):
    return np.random.uniform(size=(batch_size, n_noise))#uniform : 균등 분포 -1과 1사이에 랜덤값 추출
###############################################


G = generator(Z, Y) #노이즈를 입력받아 가짜 이미지를 생성시키는 생성자 G
D_real = discriminator(X, Y) #진짜 이미지를 구분자에 대입함
D_gene = discriminator(G, Y, True) #노이즈를 통해 얻어낸 가짜 이미지로 구분자에 대입함.

#손실함수 생성
###############################################
loss_D_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D_real, labels=tf.ones_like(D_real)))  # ones_like = 1에 가깝게
loss_D_gene = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D_gene, labels=tf.zeros_like(D_gene))) #zero_like = 0에 가깝게

# loss_D_real 과 loss_D_gene 을 더한 뒤 이 값을 최소화 하도록 최적화합니다.
loss_D = loss_D_real + loss_D_gene #loss_D가 1에 가까워질 수록 실제 이미지로 판별

# 가짜 이미지를 진짜에 가깝게 만들도록 생성망을 학습시키기 위해, D_gene 을 최대한 1에 가깝도록 만드는 손실함수입니다.
loss_G = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D_gene, labels=tf.ones_like(D_gene)))
###############################################

#변수 생성
###############################################
vars_D = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                           scope='discriminator')
vars_G = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                           scope='generator')

train_D = tf.optimizers.Adam(learning_rate).minimize(loss_D,
                                                         var_list=vars_D)
train_G = tf.optimizers.Adam(learning_rate).minimize(loss_G,
                                                         var_list=vars_G)
###############################################


#신경망 모델 학습
###############################################
sess=tf.Session()
saver = tf.train.Saver(tf.global_variables())
cpkt=tf.train.get_checkpoint_state('./GAN')  #./GAN 폴더 내에 저장한 학습 세션이 있는지 확인한다.
if cpkt and tf.train.checkpoint_exists(cpkt.model_checkpoint_path):
    saver.restore(sess,cpkt.model_checkpoint_path) #./GAN 폴더 내에 학습한 세션이 있으면 다시 불러오고 종료한다.
    print("Load sess")
else:
    sess.run(tf.global_variables_initializer()) #만약 ./GAN 폴더 내에 학습한 세션이 없었다면 초기화한다.

    for epoch in range(12): #세대 학습
        for i in range(100): #한 세대에 100회 반복하여 학습한다.
            batch_xs,batch_ys=batch(train_list, batch_size) #batch_xs는 각 이미지 별 1차원으로 된 리스트가 존재한다.
                                                            #batch_ys는 각 이미지 인덱스에 맞는 3 -class의 리스트가 존재
                                                            #자세한건 위에 기재되어있다.
            noise=get_noise(batch_size,n_noise)   #노이즈 함수를 통해 가짜 이미지와 유사하게 1차원 배열을 생성한다.

            _,loss_val_D=sess.run([train_D,loss_D],feed_dict={X:batch_xs,Y: batch_ys,Z:noise}) #구분자 학습
            _,loss_val_G=sess.run([train_G,loss_G],feed_dict={Y: batch_ys,Z:noise}) #생성자 학습

        print('Epoch:', '%04d' % epoch,
              'D loss: {:.4}'.format(loss_val_D),
              'G loss: {:.4}'.format(loss_val_G))

        if epoch==0 or (epoch)%2==0: #학습 중간 중간 학습이 잘되는지 테스트!
            sample_size=10 #이미지 10개를 확인해보기 위해 10으로 잡음
            noise = get_noise(sample_size, n_noise) #노이즈 생성
            samples=sess.run(G,feed_dict={Y: batch_ys[:sample_size],Z:noise}) #생성한 노이즈를 출력!
 
            fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 1)) #plt를 통해 이미지 출력 폼을 잡음

            for i in range(sample_size):
                ax[0][i].set_axis_off()
                ax[1][i].set_axis_off()

                ax[1][i].imshow(np.reshape(batch_xs[i], (100, 100))) #100*100 크기의 진짜 이미지 출력!
                ax[1][i].imshow(np.reshape(samples[i],(100,100))) #100*100의 노이즈로 생성한 이미지 출력

            plt.savefig('./samples2/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig) #'./samples2 폴더에 png로 저장한다!

print("End")