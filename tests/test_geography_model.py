from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from synthetic_br_profiles_gan.calibration import generate_calibration_dataset
from synthetic_br_profiles_gan.domain.geography import (
    GEO_KEY_COLUMN,
    GEOGRAPHY_MODEL_VERSION,
    GEOGRAPHY_V2_MODEL_COLUMNS,
    UNKNOWN_GEOGRAPHY_KEY,
    build_geography_catalog,
    decode_geography_key,
    encode_geography_tuple,
    geography_catalog_checksum,
    geography_key_categories,
    validate_geography_mapping,
)
from synthetic_br_profiles_gan.evaluation.geography import geography_quality_report, geography_validity_metrics
from synthetic_br_profiles_gan.evaluation.relationships import relational_validity_report
from synthetic_br_profiles_gan.metadata import MODEL_COLUMNS, default_metadata
from synthetic_br_profiles_gan.models.ctgan import CTGANSynthesizer
from synthetic_br_profiles_gan.models.programmatic import ProgrammaticSynthesizer
from synthetic_br_profiles_gan.pipeline import _postprocessing_geography_summary
from synthetic_br_profiles_gan.validators.structural import validate_profile_dataframe


class FakeCTGAN:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_data = None
        self.fit_discrete_columns = None

    def fit(self, data, discrete_columns):
        self.fit_data = data.copy()
        self.fit_discrete_columns = list(discrete_columns)

    def sample(self, rows):
        key = geography_key_categories()[0]
        return pd.DataFrame(
            {
                GEO_KEY_COLUMN: [key] * int(rows),
                "Idade": [32] * int(rows),
                "Genero": ["Feminino"] * int(rows),
                "Escolaridade": ["Ensino Médio"] * int(rows),
                "Estado_Civil": ["Solteiro"] * int(rows),
                "Ocupacao": ["Atendente"] * int(rows),
                "Renda": [2200.0] * int(rows),
                "Dependentes": [0] * int(rows),
            }
        )


class UnknownKeyCTGAN(FakeCTGAN):
    def sample(self, rows):
        frame = super().sample(rows)
        frame[GEO_KEY_COLUMN] = "GEO_999999"
        return frame


class GeographyModelTest(unittest.TestCase):
    def test_geography_catalog_is_deterministic_and_valid(self) -> None:
        first = build_geography_catalog()
        second = build_geography_catalog()
        self.assertEqual(first, second)
        self.assertEqual(len(first), validate_geography_mapping()["entries"])
        self.assertTrue(validate_geography_mapping()["is_valid"])
        self.assertEqual(len({entry.geo_key for entry in first}), len(first))
        self.assertRegex(geography_catalog_checksum(), r"^[0-9a-f]{64}$")

    def test_encode_decode_roundtrip_preserves_tuple(self) -> None:
        entry = build_geography_catalog()[0]
        key = encode_geography_tuple(entry.regiao, entry.estado, entry.municipio, entry.ddd)
        self.assertEqual(key, entry.geo_key)
        decoded = decode_geography_key(key)
        self.assertEqual(decoded, entry)

    def test_multiple_ddds_create_distinct_allowed_keys(self) -> None:
        entries = [
            entry
            for entry in build_geography_catalog()
            if entry.estado == "SP" and entry.municipio == "São Paulo"
        ]
        self.assertGreater(len(entries), 1)
        self.assertEqual(len({entry.ddd for entry in entries}), len(entries))
        self.assertEqual(len({entry.geo_key for entry in entries}), len(entries))

    def test_same_municipality_name_is_disambiguated_by_state_when_present(self) -> None:
        by_name: dict[str, list] = {}
        for entry in build_geography_catalog():
            by_name.setdefault(entry.municipio, []).append(entry)
        duplicated_names = [entries for entries in by_name.values() if len({item.estado for item in entries}) > 1]
        for entries in duplicated_names:
            keys = {
                encode_geography_tuple(entry.regiao, entry.estado, entry.municipio, entry.ddd)
                for entry in entries
            }
            self.assertEqual(len(keys), len(entries))
        if not duplicated_names:
            self.assertEqual(duplicated_names, [])

    def test_ctgan_geography_v2_trains_on_geo_key_and_exports_public_schema(self) -> None:
        metadata = default_metadata()
        train = generate_calibration_dataset(num_rows=20, seed=41)
        synthesizer = CTGANSynthesizer(
            {
                "seed": 41,
                "epochs": 1,
                "batch_size": 10,
                "verbose": False,
                "enable_gpu": False,
                "geography_model_version": GEOGRAPHY_MODEL_VERSION,
            }
        )
        with patch.object(CTGANSynthesizer, "_ctgan_class", staticmethod(lambda: FakeCTGAN)), patch(
            "synthetic_br_profiles_gan.models.ctgan.importlib.metadata.version",
            return_value="0.12.1",
        ):
            synthesizer.fit(train, metadata)
            self.assertEqual(list(synthesizer.model.fit_data.columns), GEOGRAPHY_V2_MODEL_COLUMNS)
            self.assertIn(GEO_KEY_COLUMN, synthesizer.model.fit_discrete_columns)
            sampled = synthesizer.sample(5)
        self.assertEqual(list(sampled.columns), MODEL_COLUMNS)
        self.assertNotIn(GEO_KEY_COLUMN, sampled.columns)
        self.assertEqual(geography_validity_metrics(sampled)["raw_geographic_validity_rate"], 1.0)

    def test_ctgan_geography_v2_save_load_validates_catalog(self) -> None:
        metadata = default_metadata()
        train = generate_calibration_dataset(num_rows=20, seed=42)
        synthesizer = CTGANSynthesizer(
            {
                "seed": 42,
                "epochs": 1,
                "batch_size": 10,
                "verbose": False,
                "enable_gpu": False,
                "geography_model_version": GEOGRAPHY_MODEL_VERSION,
            }
        )
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            CTGANSynthesizer,
            "_ctgan_class",
            staticmethod(lambda: FakeCTGAN),
        ), patch(
            "synthetic_br_profiles_gan.models.ctgan.importlib.metadata.version",
            return_value="0.12.1",
        ):
            path = Path(tmp)
            synthesizer.fit(train, metadata)
            synthesizer.save(path)
            loaded = CTGANSynthesizer.load(path)
            self.assertEqual(loaded.geography_model_version, GEOGRAPHY_MODEL_VERSION)
            self.assertEqual(loaded.geography_catalog_checksum, geography_catalog_checksum())
            self.assertTrue((path / "metadata_ctgan_internal.json").exists())
            self.assertTrue((path / "geography_catalog.json").exists())

    def test_unknown_geography_key_remains_invalid(self) -> None:
        synthesizer = CTGANSynthesizer({"geography_model_version": GEOGRAPHY_MODEL_VERSION})
        synthesizer.model = UnknownKeyCTGAN()
        synthesizer.metadata = default_metadata()
        synthesizer.geography_model_version = GEOGRAPHY_MODEL_VERSION
        sampled = synthesizer.sample(3)
        self.assertTrue((sampled["Regiao"] == UNKNOWN_GEOGRAPHY_KEY).all())
        self.assertTrue((sampled["DDD"] == 0).all())
        validation = validate_profile_dataframe(sampled, final=False)
        self.assertFalse(validation.report["is_valid"])

    def test_geography_report_measures_coverage_and_tvd(self) -> None:
        reference = generate_calibration_dataset(num_rows=100, seed=43)
        synthetic = reference.head(50).copy()
        report = geography_quality_report(reference, synthetic, geography_model_version=2)
        self.assertEqual(report["geography_model_version"], 2)
        self.assertEqual(report["validity"]["known_geography_key_rate"], 1.0)
        self.assertIn("geography_key_coverage", report["diversity"])
        self.assertIn("geography_key_distribution_tvd", report["diversity"])

    def test_relational_validity_reports_professional_and_non_relational_rates(self) -> None:
        frame = generate_calibration_dataset(num_rows=20, seed=44)
        report = relational_validity_report(frame)
        self.assertEqual(report["validity"]["professional_validity_rate"], 1.0)
        self.assertEqual(report["validity"]["non_relational_validity_rate"], 1.0)
        self.assertEqual(report["invalid_counts"]["professional_joint"], 0)

    def test_relational_validity_detects_invalid_professional_combination(self) -> None:
        frame = generate_calibration_dataset(num_rows=10, seed=45)
        frame.loc[0, "Escolaridade"] = "Fundamental"
        frame.loc[0, "Ocupacao"] = "Médico"
        report = relational_validity_report(frame)
        self.assertLess(report["validity"]["professional_validity_rate"], 1.0)
        self.assertEqual(report["invalid_counts"]["occupation_education"], 1)
        self.assertIn(
            {"Escolaridade": "Fundamental", "Ocupacao": "Médico", "count": 1},
            report["top_invalid_combinations"]["occupation_education"],
        )

    def test_programmatic_generation_strategy_is_unchanged(self) -> None:
        metadata = default_metadata()
        synthesizer = ProgrammaticSynthesizer({"seed": 123})
        synthesizer.fit(pd.DataFrame(columns=metadata.model_columns), metadata)
        sampled = synthesizer.sample(10)
        self.assertEqual(list(sampled.columns), MODEL_COLUMNS)
        self.assertNotIn(GEO_KEY_COLUMN, sampled.columns)

    def test_postprocessing_geography_summary_separates_repair_replacement_and_rejection(self) -> None:
        raw = pd.DataFrame(
            {
                "Regiao": ["Sudeste", "Sudeste", "Sudeste"],
                "Estado": ["SP", "SP", "SP"],
                "Municipio": ["São Paulo", "Campinas", "Niterói"],
                "DDD": [11, 21, 21],
                "Idade": [30, 30, 30],
                "Genero": ["Feminino", "Feminino", "Feminino"],
                "Escolaridade": ["Ensino Médio", "Ensino Médio", "Ensino Médio"],
                "Estado_Civil": ["Solteiro", "Solteiro", "Solteiro"],
                "Ocupacao": ["Atendente", "Atendente", "Atendente"],
                "Renda": [2000.0, 2000.0, 2000.0],
                "Dependentes": [0, 0, 0],
            }
        )
        final = raw.copy()
        final.loc[1, "DDD"] = 11
        final.loc[2, "Estado"] = "RJ"
        summary = _postprocessing_geography_summary(
            raw,
            final,
            {"reason_counts": {"municipio_estado_incompativel": 1, "ddd_estado_incompativel": 2}},
        )
        self.assertEqual(summary["geography_unchanged_rows"], 1)
        self.assertEqual(summary["geography_repaired_rows"], 1)
        self.assertEqual(summary["geography_replaced_rows"], 1)
        self.assertEqual(summary["geography_rejected_rows"], 3)
        self.assertEqual(summary["field_changes"]["DDD"]["modified_rows"], 1)
        self.assertEqual(summary["field_changes"]["Estado"]["modified_rows"], 1)


if __name__ == "__main__":
    unittest.main()
