import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

# TFHub에서 ProGAN을 임포트합니다.
module = hub.KerasLayer("https://tfhub.dev/google/progan-128/1")
# 생성할 샘플의 잠재 공간 차원
latent_dim = 512

# 시드를 바꾸면 다른 얼굴을 생성합니다.
latent_vector = tf.random.normal([1, latent_dim], seed=100)

# 모듈을 사용해 잠재 공간에서 이미지를 생성합니다.
interpolated_images = module(latent_vector)

plt.imshow(interpolated_images.numpy().reshape(128,128,3))
plt.show()
'''
def interpolate_between_vectors():
    module = hub.KerasLayer("https://tfhub.dev/google/progan-128/1")

    # 다른 랜덤 벡터를 사용하려면 시드 값을 변경하세요.
    v1 = tf.random.normal([latent_dim], seed=5)
    v2 = tf.random.normal([latent_dim], seed=1)

    # v1과 v2 사이 25개의 스텝을 담은 보간 텐서를 만듭니다.
    vectors = interpolate_hypersphere(v1, v2, 25)

    # 모듈을 사용해 잠재 공간에서 이미지를 생성합니다.
    interpolated_images = module(vectors)

    animate(interpolated_images)

interpolate_between_vectors()

image_from_module_space = True

def get_module_space_image():
    module = hub.KerasLayer("https://tfhub.dev/google/progan-128/1")
    vector = tf.random.normal([1, latent_dim], seed=30)
    images = module(vector)
    return images[0]

def upload_image():
    uploaded = files.upload()
    image = imageio.imread(uploaded[uploaded.keys()[0]])
    return transform.resize(image, [128, 128])

if image_from_module_space:
    target_image = get_module_space_image()
else:
    target_image = upload_image()
display_image(target_image)


def find_closest_latent_vector(num_optimization_steps, steps_per_image):
    images = []
    losses = []
    module = hub.KerasLayer("https://tfhub.dev/google/progan-128/1")

    initial_vector = tf.random.normal([1, latent_dim], seed=1)
    
    vector = tf.Variable(initial_vector)  
    optimizer = tf.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.losses.MeanAbsoluteError(reduction="sum")

    for step in range(num_optimization_steps):
        if (step % 100)==0:
            print()
        print('.', end='')
        with tf.GradientTape() as tape:
            image = module(vector.read_value())
            if (step % steps_per_image) == 0:
                images.append(image.numpy().reshape(128,128,3))
            target_image_difference = loss_fn(image, target_image[:,:,:3])
            # 잠재 벡터는 정규 분포에서 샘플링했습니다.
            # 잠재 벡터의 길이를 이 분포에서 얻은 벡터의 평균 길이로 제한하면 
            # 더 실제 같은 이미지를 얻을 수 있습니다.
            regularizer = tf.abs(tf.norm(vector) - np.sqrt(latent_dim))

            loss = target_image_difference + regularizer
            losses.append(loss.numpy())
        grads = tape.gradient(loss, [vector])
        optimizer.apply_gradients(zip(grads, [vector]))
    
    return images, losses

result = find_closest_latent_vector(num_optimization_steps=200, steps_per_image=5)

captions = [ f'Loss: {l:.2}' for l in result[1]]
display_images(result[0], captions)
'''