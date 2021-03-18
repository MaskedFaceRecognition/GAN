import tensorflow as tf
import tensorflow.keras as K

def upscale_layer(layer, upscale_factor):
    '''
    upscale_factor(int)만큼 층(텐서)을 업스케일합니다.
    텐서 크기는 [group, height, width, channels]입니다.
    '''
    height, width = layer.get_shape()[1:3]
    size = (upscale_factor * height, upscale_factor * width)
    upscaled_layer = tf.image.resize_nearest_neighbor(layer, size)
    return upscaled_layer

def smoothly_merge_last_layer(list_of_layers, alpha):
    '''
    임곗값 알파를 기반으로 층을 부드럽게 합칩니다.
    이 함수는 모든 층이 이미 RGB로 바뀌었다고 가정합니다.
    생성자를 위한 함수입니다.
    :list_of_layers    :   해상도(크기) 순서대로 정렬된 텐서 리스트
    :alpha             :   (0,1) 사이의 실수
    '''
    # 업스케일링을 위해 끝에서 두 번째 층을 선택합니다.
    last_fully_trained_layer = list_of_layers[-2]
    # 마지막으로 훈련된 층을 업스케일링합니다.
    last_layer_upscaled = upscale_layer(last_fully_trained_layer, 2)

    # 새로 추가된 층은 아직 완전히 훈련되지 않았습니다.
    larger_native_layer = list_of_layers[-1]

    # 합치기 전에 층 크기가 같은지 확인합니다.
    assert larger_native_layer.get_shape() == last_layer_upscaled.get_shape()

    # 곱셈은 브로드캐스팅되어 수행됩니다.
    new_layer = (1-alpha) * last_layer_upscaled + larger_native_layer * alpha

    return new_layer

def minibatch_std_layer(layer, group_size=4):
    '''
    층의 미니배치 표준편차를 계산합니다.
    층의 데이터 타입은 float32로 가정합니다. 그렇지 않으면 타입 변환이 필요합니다.
    '''
    # 미니배치는 group_size로 나눌 수 있거나 group_size 보다 같거나 작아야 합니다.
    group_size = K.backend.minimum(group_size, tf.shape(layer)[0])

    # 간단하게 쓰기 위해 크기 정보를 따로 저장합니다. 
    # 그래프 실행 전에는 일반적으로 배치 차원이 None이기 때문에 tf.shape에서 이 크기를 얻습니다.
    shape = list(K.int_shape(input))
    shape[0] = tf.shape(input)[0]

    # 미니배치 수준에서 연산하기 위해 크기를 바꿉니다. 
    # 이 코드는 층이 [그룹(G), 미니배치(M), 너비(W), 높이(H), 채널(C)]라고 가정합니다. 
    # 하지만 씨아노(Theano) 방식의 순서를 사용하는 구현도 있습니다.
    minibatch = K.backend.reshape(layer, (group_size, -1, shape[1], shape[2], shape[3]))

    # [M, W, H, C] 그룹의 평균을 계산합니다.
    minibatch -= tf.reduce_mean(minibatch, axis=0, keepdims=True)
    # [M, W, H, C] 그룹의 분산을 계산합니다.
    minibatch = tf.reduce_mean(K.backend.square(minibatch), axis = 0)
    # [M,W,H,C] 그룹의 표준편차를 계산합니다.
    minibatch = K.backend.square(minibatch + 1e-8)
    # 특성 맵을 평균하여 [M,1,1,1] 픽셀을 얻습니다.
    minibatch = tf.reduce_mean(minibatch, axis=[1,2,3], keepdims=True)
    # 스칼라 값을 그룹과 픽셀에 맞게 변환합니다.
    minibatch = K.backend.tile(minibatch, [group_size, 1, shape[2], shape[3]])
    # 새로운 특성 맵을 추가합니다.
    return K.backend.concatenate([layer, minibatch], axis=1)

def equalize_learning_rate(shape, gain, fan_in=None):
    '''
    He 초기화의 상수로 모든 층의 가중치를 조정하여 
    특성마다 각기 다른 다이나믹 레인지를 가지도록 분산을 맞춥니다.
    shape   :   텐서(층)의 크기: 각 층의 차원입니다.
        예를 들어, [4,4,48,3]. 이 경우 [커널크기, 커널크기, 필터개수, 특성맵]입니다. 
        하지만 구현에 따라 조금 다를 수 있습니다.
    gain    :   일반적으로 sqrt(2)
    fan_in  :   세이비어/He 초기화에서 입력 연결 개수
    '''

    # 기본값은 특성 맵 차원을 제외하고 shape의 모든 차원을 곱합니다. 이를 통해 뉴런마다 입력 연결 개수를 얻습니다.
    if fan_in is None: fan_in = np.prod(shape[:-1])
    # He 초기화 상수 (He et al, 2015)
    std = gain / K.sqrt(fan_in)
    # 조정을 위한 상수를 만듭니다.
    wscale = K.constant(std, name='wscale', dtype=np.float32)
    # 가중치 값을 얻어 브로드캐스팅으로 wscale을 적용합니다.
    adjusted_weights = K.get_value('layer', shape=shape, 
            initializer=tf.initializers.random_normal()) * wscale
    return adjusted_weights

def pixelwise_feat_norm(inputs, **kwargs):
    '''
    Krizhevsky 등이 2012년 논문에 제안한 픽셀별 특성 정규화
    :inputs : 케라스 / TF 층
    '''
    normalization_constant = K.backend.sqrt(K.backend.mean(
                    inputs**2, axis=-1, keepdims=True) + 1.0e-8)
    return inputs / normalization_constant


