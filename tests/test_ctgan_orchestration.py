from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_br_profiles_gan.calibration import generate_calibration_dataset
from synthetic_br_profiles_gan.metadata import default_metadata
from synthetic_br_profiles_gan.exceptions import SyntheticModelError
from synthetic_br_profiles_gan.models.ctgan import CTGANSynthesizer, _ctgan_constructor_kwargs


class FakeCTGAN:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.discrete_columns = []
        self.data = None

    def fit(self, data, discrete_columns):
        self.data = data.reset_index(drop=True)
        self.discrete_columns = list(discrete_columns)

    def sample(self, num_rows):
        return self.data.head(num_rows).copy()


class SignatureCTGAN:
    def __init__(
        self,
        *,
        epochs,
        batch_size,
        verbose,
        enable_gpu,
        embedding_dim=None,
        generator_lr=None,
        pac=10,
    ):
        pass


class CTGANOrchestrationTest(unittest.TestCase):
    def test_ctgan_declares_categorical_and_discrete_columns(self) -> None:
        fake_module = types.ModuleType("ctgan")
        fake_module.CTGAN = FakeCTGAN
        metadata = default_metadata()
        train = generate_calibration_dataset(10, seed=41)
        with patch.dict(sys.modules, {"ctgan": fake_module}):
            with patch("importlib.metadata.version", return_value="0.12.1"):
                synthesizer = CTGANSynthesizer({"epochs": 1, "batch_size": 10, "verbose": False, "cuda": False})
                synthesizer.fit(train, metadata)
                self.assertIn("DDD", synthesizer.discrete_columns)
                self.assertIn("Estado", synthesizer.discrete_columns)
                sampled = synthesizer.sample(3)
                self.assertEqual(list(sampled.columns), metadata.model_columns)
                self.assertEqual(len(sampled), 3)

    def test_ctgan_optional_kwargs_follow_installed_signature(self) -> None:
        kwargs = _ctgan_constructor_kwargs(
            SignatureCTGAN,
            {
                "epochs": 3,
                "batch_size": 100,
                "verbose": False,
                "enable_gpu": False,
                "embedding_dim": 64,
                "generator_lr": 0.0002,
                "discriminator_lr": 0.0002,
                "pac": 10,
            },
        )
        self.assertEqual(kwargs["embedding_dim"], 64)
        self.assertEqual(kwargs["generator_lr"], 0.0002)
        self.assertNotIn("discriminator_lr", kwargs)

    def test_ctgan_rejects_batch_size_not_divisible_by_pac(self) -> None:
        with self.assertRaises(SyntheticModelError):
            _ctgan_constructor_kwargs(
                SignatureCTGAN,
                {
                    "epochs": 3,
                    "batch_size": 96,
                    "verbose": False,
                    "enable_gpu": False,
                    "pac": 10,
                },
            )


if __name__ == "__main__":
    unittest.main()
