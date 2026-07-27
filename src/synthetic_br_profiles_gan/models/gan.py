"""Low-level Keras components used by the SimpleTabularGAN baseline adapter."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np

from synthetic_br_profiles_gan.exceptions import ModelBackendUnavailable

LOGGER = logging.getLogger(__name__)


def _require_tensorflow():
    try:
        from tensorflow.keras import Sequential, layers
        from tensorflow.keras.optimizers import Adam
    except ImportError as exc:
        raise ModelBackendUnavailable(
            "TensorFlow/Keras is required for SimpleTabularGAN. Install with: pip install -e \".[simple-gan]\""
        ) from exc
    return Sequential, layers, Adam


def build_generator(latent_dim: int, output_dim: int):
    """Build the dense generator used by the original notebook baseline."""
    Sequential, layers, _ = _require_tensorflow()
    return Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(output_dim, activation="sigmoid"),
        ],
        name="simple_tabular_generator",
    )


def build_discriminator(input_dim: int):
    """Build the binary discriminator for encoded tabular rows."""
    Sequential, layers, _ = _require_tensorflow()
    return Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="simple_tabular_discriminator",
    )


def build_gan(generator, discriminator, latent_dim: int, learning_rate: float = 0.0001, beta_1: float = 0.5):
    """Combine generator and discriminator into a trainable GAN."""
    Sequential, layers, Adam = _require_tensorflow()
    discriminator.trainable = False
    gan = Sequential([layers.Input(shape=(latent_dim,)), generator, discriminator], name="simple_tabular_gan")
    gan.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=learning_rate, beta_1=beta_1))
    discriminator.trainable = True
    return gan


def _loss_as_float(loss_value: Any) -> float:
    if isinstance(loss_value, (list, tuple, np.ndarray)):
        return float(np.ravel(loss_value)[0])
    return float(loss_value)


def _accuracy_as_float(loss_value: Any) -> float | None:
    if isinstance(loss_value, (list, tuple, np.ndarray)) and len(np.ravel(loss_value)) > 1:
        return float(np.ravel(loss_value)[1])
    return None


def train_gan(
    generator,
    discriminator,
    gan,
    data: np.ndarray,
    latent_dim: int,
    epochs: int = 100,
    batch_size: int = 64,
    verbose_every: int = 10,
    seed: int | None = None,
    metrics_every: int = 10,
    sample_metric_fn: Callable[[np.ndarray], dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Train the dense GAN using real epochs over all batches.

    The previous implementation used one random batch per loop and called that
    loop an epoch. This routine uses shuffled full-dataset passes and records
    batch counts, losses, discriminator accuracy, duration, seed, and optional
    fixed-sample diagnostics.
    """
    if data.size == 0:
        raise ValueError("Training data cannot be empty.")
    rng = np.random.default_rng(seed)
    history: dict[str, Any] = {
        "seed": seed,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "batches_per_epoch": int(np.ceil(data.shape[0] / batch_size)),
        "total_discriminator_updates": 0,
        "total_generator_updates": 0,
        "epochs_history": [],
    }
    fixed_noise = rng.normal(0, 1, (min(128, max(batch_size, 1)), latent_dim))
    training_start = time.perf_counter()

    for epoch in range(int(epochs)):
        epoch_start = time.perf_counter()
        permutation = rng.permutation(data.shape[0])
        discriminator_losses: list[float] = []
        discriminator_accuracies: list[float] = []
        generator_losses: list[float] = []

        for start in range(0, data.shape[0], batch_size):
            batch_indices = permutation[start : start + batch_size]
            real_samples = data[batch_indices]
            current_batch = real_samples.shape[0]
            real_labels = np.ones((current_batch, 1), dtype=np.float32)
            fake_labels = np.zeros((current_batch, 1), dtype=np.float32)

            noise = rng.normal(0, 1, (current_batch, latent_dim))
            fake_samples = generator.predict(noise, verbose=0)

            d_loss_real = discriminator.train_on_batch(real_samples, real_labels)
            d_loss_fake = discriminator.train_on_batch(fake_samples, fake_labels)
            history["total_discriminator_updates"] += 2
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            discriminator_losses.append(_loss_as_float(d_loss))
            accuracy = _accuracy_as_float(d_loss)
            if accuracy is not None:
                discriminator_accuracies.append(accuracy)

            discriminator.trainable = False
            noise = rng.normal(0, 1, (current_batch, latent_dim))
            g_loss = gan.train_on_batch(noise, real_labels)
            discriminator.trainable = True
            history["total_generator_updates"] += 1
            generator_losses.append(_loss_as_float(g_loss))

        epoch_record: dict[str, Any] = {
            "epoch": epoch + 1,
            "batches": int(np.ceil(data.shape[0] / batch_size)),
            "generator_loss": float(np.mean(generator_losses)),
            "discriminator_loss": float(np.mean(discriminator_losses)),
            "duration_seconds": float(time.perf_counter() - epoch_start),
        }
        if discriminator_accuracies:
            epoch_record["discriminator_accuracy"] = float(np.mean(discriminator_accuracies))
        if sample_metric_fn and metrics_every > 0 and ((epoch + 1) % metrics_every == 0 or epoch == 0):
            sample = generator.predict(fixed_noise, verbose=0)
            epoch_record["fixed_sample_metrics"] = sample_metric_fn(sample)
        history["epochs_history"].append(epoch_record)

        if verbose_every > 0 and ((epoch + 1) % verbose_every == 0 or epoch == 0):
            LOGGER.info(
                "simple_gan_epoch",
                extra={
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "d_loss": epoch_record["discriminator_loss"],
                    "g_loss": epoch_record["generator_loss"],
                    "d_accuracy": epoch_record.get("discriminator_accuracy"),
                },
            )

    history["duration_seconds"] = float(time.perf_counter() - training_start)
    return history
