import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def generate_latent_points(latent_dim, num_samples):
    latent = np.random.normal(size=(num_samples, latent_dim))
    return latent

def show_plot(examples, n):
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i, :, :, 0], cmap='gray_r')
    plt.savefig('test.png')	

model = load_model('generator.h5')
latent_points = generate_latent_points(100, 100)
X = model.predict(latent_points)
show_plot(X, 10)
