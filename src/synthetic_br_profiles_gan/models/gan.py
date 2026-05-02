"""Arquitetura e rotina de treinamento da GAN tabular."""

from __future__ import annotations

import numpy as np
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam


def build_generator(latent_dim: int, output_dim: int) -> Sequential:
    """Constroi o gerador tabular usado no notebook original."""
    return Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(output_dim, activation="sigmoid"),
        ],
        name="tabular_generator",
    )


def build_discriminator(input_dim: int) -> Sequential:
    """Constroi o discriminador binario para dados tabulares normalizados."""
    return Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="tabular_discriminator",
    )


def build_gan(generator: Sequential, discriminator: Sequential, latent_dim: int) -> Sequential:
    """Combina gerador e discriminador no modelo composto da GAN."""
    discriminator.trainable = False
    gan = Sequential([layers.Input(shape=(latent_dim,)), generator, discriminator], name="tabular_gan")
    gan.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0001, beta_1=0.5))
    discriminator.trainable = True
    return gan


def _loss_as_float(loss_value) -> float:
    if isinstance(loss_value, (list, tuple, np.ndarray)):
        return float(np.ravel(loss_value)[0])
    return float(loss_value)


def train_gan(
    generator: Sequential,
    discriminator: Sequential,
    gan: Sequential,
    data: np.ndarray,
    latent_dim: int,
    epochs: int = 100,
    batch_size: int = 64,
    verbose_every: int = 10,
) -> None:
    """Treina a GAN mantendo a rotina cientifica do notebook."""
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_samples = data[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        fake_samples = generator.predict(noise, verbose=0)

        d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        discriminator.trainable = False
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real_labels)
        discriminator.trainable = True

        should_log = verbose_every > 0 and ((epoch + 1) % verbose_every == 0 or epoch == 0)
        if should_log:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"D loss: {_loss_as_float(d_loss):.4f} | "
                f"G loss: {_loss_as_float(g_loss):.4f}"
            )

