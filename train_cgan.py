import tensorflow as tf
from model import conditional_gan
from config import config
from data import get_data, get_real_samples, get_fake_samples
import numpy as np
from tqdm import tqdm

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

gan = conditional_gan()
discriminator = gan.get_discriminator()
generator = gan.get_generator(config['latent_dim'])
gan_model = gan.get_model(discriminator, generator)

gan_model.load_weights('gan_model_c.h5')


# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, label=None, num_classes=10, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in tqdm(range(n_epochs)):
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            d_model.trainable = True
            # get randomly selected 'real' samples
            X_real, y_real = get_real_samples(dataset, half_batch, label=label)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = get_fake_samples(g_model, latent_dim, half_batch, label=label)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            d_model.trainable = False
            # prepare points in latent space as input for the generator
            X_gan = np.random.normal(size=(n_batch, latent_dim)) # generate_latent_points(latent_dim, n_batch)
            if label is not None:
                fake_label = np.random.randint(0, num_classes, n_batch)
                X_gan = [X_gan, fake_label]
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                    (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
    # save the generator model
    g_model.save('generator_c.h5')
    d_model.save('discriminator_c.h5')
    gan_model.save('gan_model_c.h5')

dataset = get_data('fmnist')
label = np.hstack((dataset[1], dataset[3]))
dataset = np.vstack((dataset[0], dataset[2]))
train(generator, discriminator, gan_model, dataset, config['latent_dim'], label=label, n_epochs=config['epoch'])
