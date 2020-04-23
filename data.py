import tensorflow as tf
import numpy as np

def get_data(dataset='fmnist'):
    if dataset == 'fmnist':
        # Load the fashion-mnist pre-shuffled train data and test data
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    else:
        assert 0, "dataset not support!"
    # print(f"x_train {x_train.shape}, y_train {y_train.shape}; x_test {x_test.shape}, y_test {y_test.shape}")
    return x_train, y_train, x_test, y_test

def get_real_samples(data, num_samples, label=None):
    # X = get_data(dataset)
    #X = np.vstack((data[0], data[2]))
    rand = np.random.randint(0, data.shape[0], num_samples)
    X = data[rand]
    X = np.expand_dims(X, axis=-1).astype('float32') / 255.0
    y = np.ones((num_samples, 1))
    if label is not None:
        labels = label[rand]
        return [X, labels], y
    return X, y

def get_fake_samples(generator, latent_dim, num_samples, label=None, num_classes=10):
    latent = np.random.normal(size=(num_samples, latent_dim))
    y = np.zeros((num_samples, 1))
    if label is not None:
        labels = np.random.randint(0, num_classes, num_samples)
        X = generator.predict([latent, labels])
        return [X, labels], y
    X = generator.predict(latent)
    return X, y

