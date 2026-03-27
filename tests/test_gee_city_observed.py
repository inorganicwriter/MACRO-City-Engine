from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

TEST_OUTPUT_ROOT = Path(tempfile.gettempdir()) / "macro_city_engine_test_artifacts"
os.environ["URBAN_PULSE_OUTPUT_ROOT"] = str(TEST_OUTPUT_ROOT)

from src.gee_city_observed import (
    import_gee_ghsl_city_yearly,
    import_gee_viirs_city_monthly,
    prepare_gee_city_bundle,
)


class GeeCityObservedTests(unittest.TestCase):
    def test_prepare_gee_city_bundle_writes_scripts_and_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = prepare_gee_city_bundle(
                max_cities=5,
                buffer_m=4200,
                output_dir=tmpdir,
                asset_id="users/test/gee_city_points",
                start_year=2019,
                end_year=2025,
            )
            self.assertEqual(summary["status"], "ok")
            self.assertTrue((Path(tmpdir) / "gee_city_points.csv").exists())
            self.assertTrue((Path(tmpdir) / "gee_export_viirs_monthly.js").exists())
            self.assertTrue((Path(tmpdir) / "gee_export_ghsl_yearly.js").exists())

    def test_import_gee_viirs_city_monthly_merges_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "viirs.csv"
            dest = Path(tmpdir) / "viirs_city_monthly.csv"
            pd.DataFrame(
                [
                    {"city_id": "new_york_us", "year": 2024, "month": 5, "radiance": 12.5, "cf_cvg": 7.0},
                    {"city_id": "london_gb", "year": 2024, "month": 5, "radiance": 9.8, "cf_cvg": 6.0},
                ]
            ).to_csv(source, index=False)
            pd.DataFrame(
                [
                    {
                        "city_id": "new_york_us",
                        "year": 2023,
                        "month": 7,
                        "radiance": 10.0,
                        "cf_cvg": 5.0,
                        "viirs_source": "eog_annual_v22_vcmslcfg",
                    }
                ]
            ).to_csv(dest, index=False)

            summary = import_gee_viirs_city_monthly(source_path=source, output_path=dest, merge_existing=True, max_cities=10)
            merged = pd.read_csv(dest)
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(len(merged), 3)
            self.assertIn("gee_viirs_monthly_vcmslcfg", merged["viirs_source"].astype(str).tolist())

    def test_import_gee_ghsl_city_yearly_standardizes_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "ghsl.csv"
            dest = Path(tmpdir) / "city_ghsl_yearly.csv"
            pd.DataFrame(
                [
                    {
                        "city_id": "new_york_us",
                        "year": 2020,
                        "built_surface_km2": 620.5,
                        "built_volume_total": 1_250_000_000.0,
                        "ghsl_source": "gee_ghsl_p2023a",
                    },
                    {
                        "city_id": "london_gb",
                        "year": 2020,
                        "built_surface_km2": 510.2,
                        "built_volume_total": 980_000_000.0,
                        "ghsl_source": "gee_ghsl_p2023a",
                    },
                ]
            ).to_csv(source, index=False)

            summary = import_gee_ghsl_city_yearly(source_path=source, output_path=dest, merge_existing=False, max_cities=10)
            out = pd.read_csv(dest)
            self.assertEqual(summary["status"], "ok")
            self.assertIn("ghsl_built_surface_km2", out.columns)
            self.assertIn("ghsl_built_volume_m3", out.columns)
            self.assertAlmostEqual(float(out.loc[out["city_id"] == "new_york_us", "ghsl_built_surface_km2"].iloc[0]), 620.5)


if __name__ == "__main__":
    unittest.main()
