import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose, Dropout, Input, Embedding, Concatenate
from config import config

class unconditional_gan:

    def get_discriminator(self, input_shape=(28, 28, 1)):
        model = Sequential([
            Conv2D(128, (3, 3), strides=(2, 2), padding='same', input_shape=input_shape),
            LeakyReLU(alpha=0.2),
            Conv2D(128, (3, 3), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.2),
            Flatten(),
            Dense(1, activation='sigmoid')
            ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def get_generator(self, latent_dim):
        model = Sequential([
            # 7*7
            Dense(128 * 7 * 7, input_dim=latent_dim),
            LeakyReLU(alpha=0.2),
            Reshape((7, 7, 128)),
            # 14*14
            Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.2),
            # 28*28
            Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'),
            LeakyReLU(alpha=0.2),
            # 28*28*1
            Conv2D(1, (7, 7), padding='same', activation='sigmoid')
            ])
        return model

    def get_model(self, discriminator, generator):
        # discriminator = self.get_discriminator(input_shape)
        discriminator.trainable = False
        # generator = self.get_generator(latent_dim)
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        return model

class conditional_gan:

    def get_discriminator(self, input_shape=(28, 28, 1), num_classes=10):
        # input label
        input_label = Input(shape=(1,))
        in1 = Embedding(num_classes, config['embedding_dim'])(input_label)
        in1 = Dense(input_shape[0] * input_shape[1])(in1)
        in1 = Reshape((input_shape[0], input_shape[1], 1))(in1)
        # input image
        input_image = Input(shape=input_shape)
        # merge
        x = Concatenate()([input_image, in1])
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Flatten()(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model([input_image, input_label], x)
        optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def get_generator(self, latent_dim, num_classes=10):
        # input label
        input_label = Input(shape=(1,))
        in1 = Embedding(num_classes, config['embedding_dim'])(input_label)
        in1 = Dense(7 * 7)(in1)
        in1 = Reshape((7, 7, 1))(in1)
        # input latent
        input_latent = Input(shape=(latent_dim,))
        in2 = Dense(7 * 7 * 128)(input_latent)
        in2 = LeakyReLU(alpha=0.2)(in2)
        in2 = Reshape((7, 7, 128))(in2)
        # merge
        x = Concatenate()([in2, in1])
        x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(1, (7, 7), activation='sigmoid', padding='same')(x)
        model = Model([input_latent, input_label], x)
        return model

    def get_model(self, discriminator, generator):
        discriminator.trainable = False
        latent, label = generator.input
        gen_output = generator.output
        model = Model([latent, label], discriminator([gen_output, label]))
        optimizer = tf.keras.optimizers.Adam(config['learning_rate'])
        model.compile(loss='binary_crossentropy', optimizer=optimizer)
        return model


