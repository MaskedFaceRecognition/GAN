from keras.preprocessing import image
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
img_data = image.load_img('C:\\Users\\dw\\github_repository\\senier_project_mask_to_nonMask\\gan\\photo\\size28\\train\\Ariel_Sharon\\Ariel_Sharon_6.jpg', target_size=(28,28))
img_data=np.array(img_data)
print(img_data)
import matplotlib.pyplot as plt
'''
plt.imshow(img_data)
plt.show()
'''
from keras.preprocessing.image import ImageDataGenerator
data_generator = ImageDataGenerator(rescale=1./255)

'''
train_generator = data_generator.flow_from_directory(
    'C:\\Users\\dw\\github_repository\\senier_project_mask_to_nonMask\\gan\\photo\\size28\\train\\Ariel_Sharon',
    target_size=(28, 28),
    batch_size=1,
    class_mode='categorical')

x_train, y_train = train_generator.next()
print(x_train)
print(x_train[0].shape)
plt.imshow(x_train[0])
plt.show()
'''