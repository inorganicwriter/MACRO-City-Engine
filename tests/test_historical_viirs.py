from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, TiffImagePlugin

TEST_OUTPUT_ROOT = Path(tempfile.gettempdir()) / "macro_city_engine_test_artifacts"
os.environ["URBAN_PULSE_OUTPUT_ROOT"] = str(TEST_OUTPUT_ROOT)

from src.historical_viirs import _parse_temporal_meta, _sample_geotiff_city_values, merge_viirs_monthly_panels


def _write_geotiff(path: Path, arr: np.ndarray) -> None:
    img = Image.fromarray(arr.astype("float32"), mode="F")
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33550] = (1.0, 1.0, 0.0)
    ifd[33922] = (0.0, 0.0, 0.0, 0.0, 10.0, 0.0)
    img.save(path, format="TIFF", tiffinfo=ifd)


class HistoricalViirsTests(unittest.TestCase):
    def test_parse_temporal_meta_monthly_and_annual(self) -> None:
        monthly = Path("VNL_v2_npp_2019-03_global_vcmslcfg_c202104010000.average.tif")
        annual = Path("VNL_v2_npp_2020_global_vcmslcfg_c202102150000.average.tif.gz")

        meta_m = _parse_temporal_meta(monthly)
        meta_a = _parse_temporal_meta(annual)

        self.assertIsNotNone(meta_m)
        self.assertIsNotNone(meta_a)
        self.assertEqual(meta_m["temporal"], "monthly")
        self.assertEqual(meta_m["year"], 2019)
        self.assertEqual(meta_m["month"], 3)
        self.assertEqual(meta_a["temporal"], "annual")
        self.assertEqual(meta_a["year"], 2020)

    def test_sample_geotiff_city_values(self) -> None:
        arr = np.arange(100, dtype=np.float32).reshape(10, 10)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "VNL_v2_npp_2019-03_global_vcmslcfg_c202104010000.average.tif"
            _write_geotiff(path, arr)

            pts = [
                {"city_id": "city_a", "city_name": "A", "sample_id": 0, "lat": 9.0, "lon": 1.0},
                {"city_id": "city_a", "city_name": "A", "sample_id": 1, "lat": 9.0, "lon": 2.0},
                {"city_id": "city_b", "city_name": "B", "sample_id": 0, "lat": 5.0, "lon": 4.0},
            ]
            sampled = _sample_geotiff_city_values(path, pts)

            self.assertIn("city_a", sampled)
            self.assertIn("city_b", sampled)
            self.assertAlmostEqual(sampled["city_a"], 11.5, places=4)
            self.assertAlmostEqual(sampled["city_b"], 54.0, places=4)

    def test_merge_viirs_monthly_panels_prefers_monthly_over_annual(self) -> None:
        existing = pd.DataFrame(
            [
                {
                    "city_id": "city_a",
                    "year": 2020,
                    "month": 7,
                    "radiance": 50.0,
                    "cf_cvg": np.nan,
                    "viirs_source": "eog_annual_v22_vcmslcfg",
                }
            ]
        )
        imported = pd.DataFrame(
            [
                {
                    "city_id": "city_a",
                    "year": 2020,
                    "month": 7,
                    "radiance": 80.0,
                    "cf_cvg": 12.0,
                    "viirs_source": "eog_monthly_v10_vcmcfg",
                }
            ]
        )
        merged = merge_viirs_monthly_panels(existing, imported)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(float(merged["radiance"].iloc[0]), 80.0)
        self.assertAlmostEqual(float(merged["cf_cvg"].iloc[0]), 12.0)
        self.assertEqual(str(merged["viirs_source"].iloc[0]), "eog_monthly_v10_vcmcfg")


if __name__ == "__main__":
    unittest.main()
