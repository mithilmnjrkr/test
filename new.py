from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.layers import Conv2D, Conv2DTranspose
from keras.optimizers import Adam
import numpy as np

# Generator model
generator = Sequential([
    Dense(128, input_shape=(100,), activation='relu'),
    Dense(256, activation='relu'),
    Dense(512, activation='relu'),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28, 1))
])

# Discriminator model
discriminator = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Combined model (GAN)
discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

# Train the GAN (Code for training not included here)
