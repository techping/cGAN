import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from model import conditional_gan
from config import config

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def generate_latent_points(latent_dim, num_samples, num_classes=10):
    latent = np.random.normal(size=(num_samples, latent_dim))
    labels = np.random.randint(0, num_classes, num_samples)
    return [latent, labels]

def save_plot(examples, n):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    plt.savefig('cgan_result.png')

gan = conditional_gan()
model = gan.get_generator(config['latent_dim'])
latent_points, labels = generate_latent_points(100, 100)
labels = np.array([x for _ in range(10) for x in range(10)])
X = model.predict([latent_points, labels])
save_plot(X, 10)
