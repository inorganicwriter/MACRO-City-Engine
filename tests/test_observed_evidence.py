from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_OUTPUT_ROOT = Path(tempfile.gettempdir()) / "urban_pulse_test_artifacts"
os.environ["URBAN_PULSE_OUTPUT_ROOT"] = str(TEST_OUTPUT_ROOT)

from src.observed_evidence import run_observed_evidence_suite
from src.utils import DATA_OUTPUTS


class ObservedEvidenceTests(unittest.TestCase):
    def test_observed_evidence_suite_runs(self) -> None:
        output_paths = [
            DATA_OUTPUTS / "observed_measurement_audit.csv",
            DATA_OUTPUTS / "observed_measurement_summary.json",
            DATA_OUTPUTS / "observed_feature_group_ablation.csv",
            DATA_OUTPUTS / "observed_feature_group_summary.csv",
            DATA_OUTPUTS / "observed_feature_group_summary.json",
            DATA_OUTPUTS / "observed_cross_source_city_year.csv",
            DATA_OUTPUTS / "observed_cross_source_pairs.csv",
            DATA_OUTPUTS / "observed_cross_source_summary.json",
            DATA_OUTPUTS / "observed_evidence_summary.json",
        ]
        backups: dict[Path, bytes | None] = {}
        for path in output_paths:
            backups[path] = path.read_bytes() if path.exists() else None

        rows = []
        city_count = 15
        years = list(range(2019, 2024))
        for city_idx in range(city_count):
            city_id = f"city_{city_idx:02d}"
            base = 40.0 + city_idx
            for year in years:
                t = year - years[0]
                rows.append(
                    {
                        "city_id": city_id,
                        "city_name": f"City {city_idx}",
                        "country": "Testland",
                        "continent": "Test",
                        "iso3": "TST",
                        "year": year,
                        "index_spec_version": "city_observed_primary_v3_with_sentiment",
                        "macro_resolution_level": "country_year",
                        "city_macro_observed_flag": 0,
                        "road_source": "osm_overpass_snapshot",
                        "viirs_source": "noaa_viirs_nightly_radiance",
                        "osm_hist_source": "osm_history_extract",
                        "poi_source": "osm_snapshot",
                        "weather_source": "open_meteo_archive",
                        "social_sentiment_source": "missing",
                        "social_sentiment_volume": 0.0,
                        "latitude": 10.0 + city_idx * 0.2,
                        "longitude": 20.0 + city_idx * 0.2,
                        "temperature_mean": 15.0 + t * 0.5,
                        "precipitation_sum": 800.0 + city_idx * 5.0,
                        "climate_comfort": 0.6 + 0.01 * t,
                        "amenity_ratio": 0.35 + 0.005 * city_idx,
                        "commerce_ratio": 0.28 + 0.01 * t,
                        "transport_intensity": 0.22 + 0.004 * city_idx,
                        "poi_diversity": 0.55 + 0.01 * t,
                        "observed_activity_signal": base + 1.5 * t,
                        "observed_mobility_signal": base + 1.2 * t,
                        "observed_livability_signal": base + 0.8 * t,
                        "observed_innovation_signal": base + 1.0 * t,
                        "observed_dynamic_signal": 45.0 + 1.1 * t + 0.2 * city_idx,
                        "observed_sentiment_signal": 50.0,
                        "road_access_score": 55.0 + city_idx,
                        "road_tier_code": int(city_idx % 3),
                        "road_length_km_total": 100.0 + 3.0 * city_idx + 2.0 * t,
                        "arterial_share": 0.30 + 0.01 * (city_idx % 4),
                        "intersection_density": 12.0 + 0.5 * city_idx,
                        "viirs_ntl_mean": 30.0 + 0.8 * city_idx + 0.5 * t,
                        "viirs_ntl_p90": 40.0 + 0.9 * city_idx + 0.6 * t,
                        "viirs_lit_area_km2": 15.0 + 0.4 * city_idx + 0.2 * t,
                        "osm_hist_road_length_m": 120000.0 + 800.0 * city_idx + 500.0 * t,
                        "osm_hist_building_count": 2000.0 + 30.0 * city_idx + 10.0 * t,
                        "osm_hist_poi_count": 600.0 + 15.0 * city_idx + 8.0 * t,
                        "road_arterial_growth_proxy": 0.03 * t,
                        "road_local_growth_proxy": 0.02 * t,
                        "road_growth_intensity": 0.10 * (t - 1.5) + 0.01 * city_idx,
                        "viirs_ntl_yoy": 0.20 * (t - 1.0) + 0.02 * city_idx,
                        "osm_hist_road_yoy": 0.12 * (t - 1.0) + 0.01 * city_idx,
                        "osm_hist_building_yoy": 0.08 * (t - 1.0) + 0.005 * city_idx,
                        "osm_hist_poi_yoy": 0.09 * (t - 1.0) + 0.006 * city_idx,
                        "osm_hist_poi_food_yoy": 0.05 * (t - 1.0) + 0.004 * city_idx,
                        "osm_hist_poi_retail_yoy": 0.06 * (t - 1.0) + 0.004 * city_idx,
                        "osm_hist_poi_nightlife_yoy": 0.04 * (t - 1.0) + 0.003 * city_idx,
                        "gdp_per_capita": 10000.0 + 120.0 * city_idx + 80.0 * t,
                        "population": 500000.0 + 10000.0 * city_idx,
                        "unemployment": 7.0 - 0.1 * t + 0.02 * city_idx,
                        "internet_users": 70.0 + 0.5 * city_idx + 0.4 * t,
                        "capital_formation": 20.0 + 0.2 * city_idx,
                        "inflation": 2.0 + 0.1 * t,
                        "employment_rate": 60.0 + 0.3 * city_idx + 0.2 * t,
                        "urban_population_share": 65.0 + 0.1 * city_idx,
                        "electricity_access": 98.0,
                        "fixed_broadband_subscriptions": 25.0 + 0.2 * city_idx,
                        "pm25_exposure": 18.0 - 0.1 * t,
                        "policy_event_count_iso_year": float((city_idx + t) % 3),
                        "policy_event_type_count_iso_year": float((city_idx + t) % 2 + 1),
                        "policy_event_new_count_iso_year": float((city_idx + 2 * t) % 2),
                        "policy_intensity_sum_iso_year": 1.5 + 0.1 * t,
                        "policy_intensity_mean_iso_year": 0.7 + 0.05 * t,
                        "policy_event_count_iso_year_yoy": 0.1 * (t - 1.0),
                        "policy_intensity_sum_iso_year_yoy": 0.08 * (t - 1.0),
                        "policy_news_proxy_score": 0.4 + 0.03 * t,
                        "social_sentiment_score": np.nan,
                        "social_sentiment_buzz": 0.0,
                        "social_sentiment_delta_1": 0.0,
                    }
                )
        panel = pd.DataFrame(rows)
        panel["economic_vitality"] = 0.45 * panel["observed_activity_signal"] + 0.15 * panel["road_access_score"]
        panel["livability"] = 0.55 * panel["observed_livability_signal"] + 0.10 * panel["climate_comfort"] * 100.0
        panel["innovation"] = 0.50 * panel["observed_innovation_signal"] + 0.20 * panel["osm_hist_poi_yoy"] * 100.0
        panel["composite_index"] = (
            0.4 * panel["economic_vitality"] + 0.35 * panel["livability"] + 0.25 * panel["innovation"]
        )

        try:
            summary = run_observed_evidence_suite(panel)

            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["measurement_audit"]["observed_primary_spec_share"], 1.0)
            self.assertEqual(summary["feature_group_ablation"]["status"], "ok")
            self.assertGreaterEqual(summary["feature_group_ablation"]["targets_evaluated"], 4)
            self.assertEqual(summary["cross_source_consistency"]["status"], "ok")
            self.assertGreaterEqual(len(summary["cross_source_consistency"]["signals_used"]), 2)
        finally:
            for path, payload in backups.items():
                if payload is None:
                    if path.exists():
                        path.unlink()
                else:
                    path.write_bytes(payload)


if __name__ == "__main__":
    unittest.main()
