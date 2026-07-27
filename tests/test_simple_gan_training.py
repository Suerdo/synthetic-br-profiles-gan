from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from synthetic_br_profiles_gan.models.gan import train_gan


class FakeGenerator:
    def __init__(self) -> None:
        self.predict_calls = 0

    def predict(self, noise, verbose=0):
        self.predict_calls += 1
        return np.zeros((noise.shape[0], 3), dtype=np.float32)


class FakeDiscriminator:
    def __init__(self) -> None:
        self.trainable = True
        self.train_calls = 0

    def train_on_batch(self, samples, labels):
        self.train_calls += 1
        return [0.5, 0.75]


class FakeGAN:
    def __init__(self) -> None:
        self.train_calls = 0

    def train_on_batch(self, noise, labels):
        self.train_calls += 1
        return 0.25


class SimpleGANTrainingTest(unittest.TestCase):
    def test_epochs_cover_all_batches_and_record_updates(self) -> None:
        data = np.ones((5, 3), dtype=np.float32)
        generator = FakeGenerator()
        discriminator = FakeDiscriminator()
        gan = FakeGAN()
        history = train_gan(
            generator=generator,
            discriminator=discriminator,
            gan=gan,
            data=data,
            latent_dim=2,
            epochs=2,
            batch_size=3,
            verbose_every=0,
            metrics_every=0,
            seed=41,
        )
        self.assertEqual(history["batches_per_epoch"], 2)
        self.assertEqual([epoch["batches"] for epoch in history["epochs_history"]], [2, 2])
        self.assertEqual(history["total_generator_updates"], 4)
        self.assertEqual(history["total_discriminator_updates"], 8)
        self.assertEqual(discriminator.train_calls, 8)
        self.assertEqual(gan.train_calls, 4)

    def test_dataset_smaller_than_batch_has_one_batch(self) -> None:
        history = train_gan(
            generator=FakeGenerator(),
            discriminator=FakeDiscriminator(),
            gan=FakeGAN(),
            data=np.ones((2, 3), dtype=np.float32),
            latent_dim=2,
            epochs=1,
            batch_size=10,
            verbose_every=0,
            metrics_every=0,
            seed=41,
        )
        self.assertEqual(history["batches_per_epoch"], 1)
        self.assertEqual(history["total_generator_updates"], 1)
        self.assertEqual(history["total_discriminator_updates"], 2)


if __name__ == "__main__":
    unittest.main()
