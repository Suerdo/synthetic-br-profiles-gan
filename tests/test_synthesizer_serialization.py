from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from synthetic_br_profiles_gan.calibration import generate_calibration_dataset
from synthetic_br_profiles_gan.exceptions import SyntheticModelError
from synthetic_br_profiles_gan.metadata import default_metadata
from synthetic_br_profiles_gan.models.programmatic import ProgrammaticSynthesizer


RUN_SLOW_MODELS = os.environ.get("RUN_SLOW_MODEL_TESTS") == "1"


class SynthesizerSerializationTest(unittest.TestCase):
    def assert_contract(self, synthesizer, train: pd.DataFrame) -> None:
        metadata = default_metadata()
        synthesizer.fit(train, metadata)
        first = synthesizer.sample(4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            synthesizer.save(path)
            loaded = type(synthesizer).load(path)
            second = loaded.sample(4)
        self.assertEqual(list(first.columns), metadata.model_columns)
        self.assertEqual(list(second.columns), metadata.model_columns)
        self.assertEqual(loaded.metadata.to_dict()["model_columns"], metadata.model_columns)
        self.assertFalse(second.isna().any().any())

    def test_programmatic_save_load_contract(self) -> None:
        train = generate_calibration_dataset(20, seed=41)
        self.assert_contract(ProgrammaticSynthesizer({"seed": 123}), train)

    def test_unfitted_programmatic_save_is_allowed_because_no_fit_state_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            synthesizer = ProgrammaticSynthesizer({"seed": 123})
            synthesizer.save(Path(tmp))
            loaded = ProgrammaticSynthesizer.load(Path(tmp))
            self.assertEqual(list(loaded.sample(2).columns), default_metadata().model_columns)

    @unittest.skipUnless(RUN_SLOW_MODELS and importlib.util.find_spec("tensorflow"), "set RUN_SLOW_MODEL_TESTS=1 with TensorFlow")
    def test_simple_gan_save_load_contract_optional(self) -> None:
        from synthetic_br_profiles_gan.models.simple_gan import SimpleTabularGAN

        train = generate_calibration_dataset(16, seed=7)
        self.assert_contract(
            SimpleTabularGAN({"seed": 7, "epochs": 1, "batch_size": 8, "latent_dim": 4, "verbose_every": 0, "metrics_every": 0}),
            train,
        )

    @unittest.skipUnless(RUN_SLOW_MODELS and importlib.util.find_spec("ctgan"), "set RUN_SLOW_MODEL_TESTS=1 with CTGAN")
    def test_ctgan_save_load_contract_optional(self) -> None:
        from synthetic_br_profiles_gan.models.ctgan import CTGANSynthesizer

        train = generate_calibration_dataset(30, seed=8)
        self.assert_contract(
            CTGANSynthesizer({"seed": 8, "epochs": 1, "batch_size": 10, "verbose": False, "enable_gpu": False}),
            train,
        )

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"), "TensorFlow not installed")
    def test_simple_gan_unfitted_sample_raises_specific_error(self) -> None:
        from synthetic_br_profiles_gan.models.simple_gan import SimpleTabularGAN

        with self.assertRaises(SyntheticModelError):
            SimpleTabularGAN({"epochs": 1}).sample(1)


if __name__ == "__main__":
    unittest.main()
