from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_OUTPUT_ROOT = Path(tempfile.gettempdir()) / "urban_pulse_test_artifacts"
os.environ["URBAN_PULSE_OUTPUT_ROOT"] = str(TEST_OUTPUT_ROOT)

from src.config import load_config
from src.dynamic_causal_envelope import _centered_pre_abs_mean
from src.econometrics import _build_timing_ready_panel, run_did_two_way_fe, run_identification_scorecard, run_matrix_completion_counterfactual
import src.global_data as gd
import src.modeling as modeling
from src.global_data import (
    _apply_policy_design_from_registry,
    _reconstruct_historical_poi_from_snapshot,
    _weighted_row_blend,
)
from src.identification_plus import _build_event_pretrend_geometry
from src.utils import DATA_OUTPUTS


class TestRegressionFixes(unittest.TestCase):
    def test_unified_dashboard_template_is_valid_html(self) -> None:
        template = Path("web/templates/dashboard_unified.html")
        text = template.read_text(encoding="utf-8")
        self.assertTrue(text.lstrip().lower().startswith("<!doctype html>"))

        required_markers = [
            'id="year-select"',
            'id="continent-select"',
            'id="refresh-btn"',
            'id="k-gate"',
            'id="nowcast-chart"',
            'id="gate-chart"',
            'id="quadrant-chart"',
            'id="evidence-chart"',
            'id="risk-body"',
            'id="summary-body"',
            'id="narrative-text"',
            'id="actions-list"',
        ]
        for marker in required_markers:
            self.assertIn(marker, text)

    def test_main_treatment_track_prefers_direct_core_when_available(self) -> None:
        cfg = load_config()
        years = [2019, 2020, 2021]
        rows = []
        for year in years:
            rows.append({"city_id": "city_aaa", "iso3": "AAA", "year": year})
            rows.append({"city_id": "city_bbb", "iso3": "BBB", "year": year})
        panel = pd.DataFrame(rows)

        registry = pd.DataFrame(
            [
                {
                    "iso3": "AAA",
                    "start_year": 2020,
                    "end_year": 2020,
                    "policy_intensity": 0.90,
                    "policy_name": "direct_policy",
                    "source_ref": "wb_project:P123456",
                },
                {
                    "iso3": "BBB",
                    "start_year": 2019,
                    "end_year": 2019,
                    "policy_intensity": 0.85,
                    "policy_name": "macro_rule_policy",
                    "source_ref": "objective_macro_rule:capital_formation",
                },
            ]
        )

        out, meta = _apply_policy_design_from_registry(panel, registry, cfg)

        aaa = out[out["iso3"] == "AAA"].copy()
        bbb = out[out["iso3"] == "BBB"].copy()

        self.assertEqual(meta.get("principal_treatment_track"), "direct_core")
        self.assertTrue((aaa["treated_city"] == 1).all())
        self.assertTrue((bbb["treated_city"] == 0).all())
        self.assertTrue((bbb["treated_city_all_sources"] == 1).all())
        self.assertEqual(int(bbb["did_treatment"].sum()), 0)

    def test_timing_ready_panel_restricts_to_supported_post_reference_cohorts(self) -> None:
        rows = []
        cohort_map = {"a": 2017, "b": 2020, "c": 2022, "d": 9999}
        for city_id, cohort in cohort_map.items():
            for year in range(2018, 2026):
                treated = int(cohort < 9999)
                post = int(treated == 1 and year >= cohort)
                rows.append(
                    {
                        "city_id": city_id,
                        "year": year,
                        "treated_city": treated,
                        "treatment_cohort_year": cohort,
                        "post_policy": post,
                        "did_treatment": treated * post,
                    }
                )
        panel = pd.DataFrame(rows)
        out, meta = _build_timing_ready_panel(panel, treatment_year=2020, window_pre=2, window_post=2, min_treated_cities_per_cohort=1)
        self.assertEqual(meta.get("status"), "ok")
        self.assertEqual(meta.get("kept_cohorts"), [2020, 2022])
        kept = out.loc[out["treated_city"] == 1, ["city_id", "treatment_cohort_year"]].drop_duplicates()
        self.assertEqual(sorted(kept["city_id"].tolist()), ["b", "c"])

    def test_identification_scorecard_falls_back_when_robust_nyt_is_nan(self) -> None:
        summary = {
            "not_yet_treated_did": {
                "status": "ok",
                "cohort_count": 2,
                "att_weighted": -0.4,
                "p_value_weighted": 0.04,
                "ci95_weighted": [-0.7, -0.1],
                "robust_att_weighted": float("nan"),
                "robust_p_value_weighted": float("nan"),
                "robust_ci95_weighted": [None, None],
                "placebo_share_p_lt_0_10": 0.0,
                "placebo_max_abs_t": 1.0,
            }
        }
        out = run_identification_scorecard(summary)
        rows = out.get("ranking", [])
        nyt = next((row for row in rows if row.get("estimator") == "not_yet_treated_did"), None)
        self.assertIsNotNone(nyt)
        self.assertAlmostEqual(float(nyt["effect"]), -0.4, places=6)
        self.assertAlmostEqual(float(nyt["p_value"]), 0.04, places=6)

    def test_pretrend_geometry_allows_two_pre_period_magnitude_check(self) -> None:
        event_path = DATA_OUTPUTS / "econometric_source_event_study_points.csv"
        econ_path = DATA_OUTPUTS / "econometric_summary.json"
        bak_event = event_path.read_bytes() if event_path.exists() else None
        bak_econ = econ_path.read_bytes() if econ_path.exists() else None
        try:
            pd.DataFrame(
                [
                    {"variant": "source_a", "design_variant": "s_a", "source_channel": "a", "rel_year": -3, "coef": 0.23, "t_value": 1.5},
                    {"variant": "source_a", "design_variant": "s_a", "source_channel": "a", "rel_year": -2, "coef": 0.27, "t_value": 1.9},
                    {"variant": "source_a", "design_variant": "s_a", "source_channel": "a", "rel_year": 0, "coef": 0.31, "t_value": 2.2},
                    {"variant": "source_a", "design_variant": "s_a", "source_channel": "a", "rel_year": 1, "coef": 0.28, "t_value": 2.0},
                ]
            ).to_csv(event_path, index=False)
            econ_path.write_text("{}", encoding="utf-8")
            out = _build_event_pretrend_geometry()
            row = out[out["variant"] == "source_a"].iloc[0]
            self.assertEqual(int(row["n_pre"]), 2)
            self.assertEqual(int(row["pretrend_pass"]), 1)
        finally:
            if bak_event is None:
                if event_path.exists():
                    event_path.unlink()
            else:
                event_path.write_bytes(bak_event)
            if bak_econ is None:
                if econ_path.exists():
                    econ_path.unlink()
            else:
                econ_path.write_bytes(bak_econ)

    def test_centered_pre_abs_mean_uses_pre_demeaned_noise(self) -> None:
        event = pd.DataFrame(
            [
                {"rel_year": -4, "coef_weighted_mean": 0.31, "post_period": 0},
                {"rel_year": -3, "coef_weighted_mean": 0.23, "post_period": 0},
                {"rel_year": -2, "coef_weighted_mean": 0.27, "post_period": 0},
                {"rel_year": 0, "coef_weighted_mean": 0.30, "post_period": 1},
            ]
        )
        raw_abs = float(np.mean(np.abs(event.loc[event["post_period"] == 0, "coef_weighted_mean"].to_numpy(dtype=float))))
        centered_abs = _centered_pre_abs_mean(event)
        expected = float(np.mean(np.abs(np.array([0.31, 0.23, 0.27]) - np.mean([0.31, 0.23, 0.27]))))
        self.assertLess(centered_abs, raw_abs)
        self.assertAlmostEqual(centered_abs, expected, places=8)

    def test_matrix_completion_uses_city_specific_cohorts(self) -> None:
        rows = []
        for city_id, cohort, base in [
            ("treat_early", 2020, 10.0),
            ("treat_late", 2022, 11.5),
            ("ctrl_a", 9999, 9.0),
            ("ctrl_b", 9999, 8.7),
            ("ctrl_c", 9999, 9.3),
            ("ctrl_d", 9999, 8.9),
            ("ctrl_e", 9999, 9.1),
            ("ctrl_f", 9999, 8.8),
        ]:
            for year in range(2018, 2026):
                treated = int(cohort < 9999)
                post = int(treated == 1 and year >= cohort)
                level = base + 0.25 * (year - 2018) + (0.6 if post and city_id == "treat_early" else 0.0)
                rows.append(
                    {
                        "city_id": city_id,
                        "year": year,
                        "treated_city": treated,
                        "treatment_cohort_year": cohort,
                        "composite_index": level,
                    }
                )
        panel = pd.DataFrame(rows)
        out = run_matrix_completion_counterfactual(panel, treatment_year=2020, placebo_count=4, random_state=7)
        self.assertEqual(out.get("status"), "ok")
        self.assertEqual(out.get("cohort_mode"), "city_specific")
        self.assertEqual(int(out.get("cohort_count", 0)), 2)
        self.assertGreater(int(out.get("evaluated_post_cells", 0)), 0)
        self.assertTrue((DATA_OUTPUTS / "matrix_completion_by_cohort.csv").exists())

    def test_weighted_row_blend_reweights_missing_components(self) -> None:
        out = _weighted_row_blend(
            [
                (0.7, pd.Series([1.0, np.nan])),
                (0.3, pd.Series([0.0, 0.5])),
            ],
            default=np.nan,
        )
        self.assertAlmostEqual(float(out.iloc[0]), 0.7, places=8)
        self.assertAlmostEqual(float(out.iloc[1]), 0.5, places=8)

    def test_poi_backcast_uses_osm_history_and_blocks_lookahead(self) -> None:
        cfg = load_config()
        panel = pd.DataFrame(
            {
                "city_id": ["a", "a", "b", "b"],
                "year": [2024, 2025, 2024, 2025],
                "amenity_count": [10, 10, 20, 20],
                "shop_count": [5, 5, 8, 8],
                "office_count": [2, 2, 3, 3],
                "leisure_count": [1, 1, 2, 2],
                "transport_count": [2, 2, 4, 4],
                "poi_total": [20, 20, 37, 37],
                "poi_diversity": [1.0, 1.0, 1.1, 1.1],
                "osm_hist_poi_count": [8.0, 10.0, np.nan, 12.0],
            }
        )
        out = _reconstruct_historical_poi_from_snapshot(panel, cfg)
        a_2024 = out[(out["city_id"] == "a") & (out["year"] == 2024)].iloc[0]
        b_2024 = out[(out["city_id"] == "b") & (out["year"] == 2024)].iloc[0]
        self.assertAlmostEqual(float(a_2024["amenity_count"]), 8.0, places=8)
        self.assertEqual(str(a_2024["poi_temporal_source"]), "snapshot_backcast_from_osm_history")
        self.assertTrue(pd.isna(b_2024["amenity_count"]))
        self.assertEqual(str(b_2024["poi_temporal_source"]), "missing_prevented_lookahead")

    def test_collect_city_viirs_year_panel_sets_observation_flags(self) -> None:
        cfg = load_config()
        cities = pd.DataFrame(
            [
                {
                    "city_id": "city_a",
                    "city_name": "City A",
                    "iso3": "AAA",
                    "continent": "Asia",
                    "latitude": 0.0,
                    "longitude": 0.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pd.DataFrame(
                [
                    {"city_id": "city_a", "year": 2025, "month": 6, "radiance": 10.0, "lit_area_km2": 2.0},
                    {"city_id": "city_a", "year": 2025, "month": 7, "radiance": 12.0, "lit_area_km2": 2.4},
                ]
            ).to_csv(tmp_path / "viirs_city_monthly.csv", index=False)
            bak_raw = gd.DATA_RAW
            gd.DATA_RAW = tmp_path
            try:
                out = gd.collect_city_viirs_year_panel(cities, cfg, use_cache=True)
            finally:
                gd.DATA_RAW = bak_raw

        row_2025 = out[out["year"] == 2025].iloc[0]
        row_2024 = out[out["year"] == 2024].iloc[0]
        self.assertEqual(int(row_2025["has_viirs_observation"]), 1)
        self.assertAlmostEqual(float(row_2025["viirs_month_count"]), 2.0, places=8)
        self.assertAlmostEqual(float(row_2025["viirs_year_coverage_share"]), 2.0 / 12.0, places=8)
        self.assertEqual(int(row_2024["has_viirs_observation"]), 0)

    def test_collect_city_social_sentiment_panel_supports_gdelt_yearly(self) -> None:
        cfg = load_config()
        cities = pd.DataFrame(
            [
                {
                    "city_id": "city_a",
                    "city_name": "City A",
                    "iso3": "AAA",
                    "continent": "Asia",
                    "latitude": 0.0,
                    "longitude": 0.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pd.DataFrame(
                [
                    {"city_id": "city_a", "year": 2025, "avg_tone": 25.0, "num_articles": 10},
                ]
            ).to_csv(tmp_path / "gdelt_city_yearly.csv", index=False)
            bak_raw = gd.DATA_RAW
            gd.DATA_RAW = tmp_path
            try:
                out = gd.collect_city_social_sentiment_panel(cities, cfg, use_cache=True)
            finally:
                gd.DATA_RAW = bak_raw

        row_2025 = out[out["year"] == 2025].iloc[0]
        self.assertAlmostEqual(float(row_2025["social_sentiment_score"]), 0.25, places=8)
        self.assertAlmostEqual(float(row_2025["social_sentiment_positive_share"]), 0.625, places=8)
        self.assertAlmostEqual(float(row_2025["social_sentiment_buzz"]), np.log1p(10.0), places=8)
        self.assertEqual(str(row_2025["social_sentiment_source"]), "gdelt_city_yearly")

    def test_engineer_features_preserves_existing_extra_wb_source(self) -> None:
        cfg = load_config()
        panel = pd.DataFrame(
            {
                "city_id": ["city_a"],
                "city_name": ["City A"],
                "iso3": ["AAA"],
                "country": ["Aland"],
                "continent": ["Asia"],
                "year": [2025],
                "macro_source": ["world_bank"],
                "extra_wb_source": ["world_bank"],
                "weather_source": ["open-meteo"],
                "poi_source": ["osm"],
                "city_macro_observed_flag": [0],
                "macro_observed_source": ["missing"],
                "macro_resolution_level": ["country_year"],
                "gdp_per_capita": [10000.0],
                "population": [1_000_000.0],
                "unemployment": [5.0],
                "internet_users": [70.0],
                "capital_formation": [25.0],
                "inflation": [2.0],
                "patent_residents": [10.0],
                "researchers_per_million": [500.0],
                "high_tech_exports_share": [15.0],
                "employment_rate": [60.0],
                "urban_population_share": [70.0],
                "electricity_access": [98.0],
                "fixed_broadband_subscriptions": [25.0],
                "pm25_exposure": [20.0],
                "temperature_mean": [18.0],
                "precipitation_sum": [800.0],
                "amenity_count": [100.0],
                "shop_count": [50.0],
                "office_count": [20.0],
                "leisure_count": [30.0],
                "transport_count": [15.0],
                "poi_total": [215.0],
                "poi_diversity": [1.1],
                "road_length_km_total": [100.0],
                "arterial_share": [0.3],
                "intersection_density": [5.0],
                "viirs_ntl_mean": [10.0],
                "viirs_ntl_p90": [12.0],
                "viirs_lit_area_km2": [5.0],
                "viirs_log_mean": [np.log1p(10.0)],
                "viirs_intra_year_recovery": [0.1],
                "viirs_intra_year_decline": [0.0],
                "viirs_recent_drop": [0.0],
                "viirs_ntl_yoy": [1.0],
                "viirs_month_count": [12.0],
                "has_viirs_observation": [1],
                "viirs_year_coverage_share": [1.0],
                "social_sentiment_score": [0.1],
                "social_sentiment_volatility": [0.05],
                "social_sentiment_positive_share": [0.55],
                "social_sentiment_negative_share": [0.45],
                "social_sentiment_volume": [100.0],
                "social_sentiment_platform_count": [1.0],
                "social_sentiment_buzz": [np.log1p(100.0)],
                "social_sentiment_source": ["gdelt_city_yearly"],
            }
        )

        out = gd._engineer_features(
            panel,
            cfg,
            add_idiosyncratic_noise=False,
            enable_city_macro_disaggregation=False,
            normalize_within_year=False,
            prefer_pca_composite=True,
        )
        self.assertEqual(str(out.loc[0, "extra_wb_source"]), "world_bank")

    def test_city_macro_observed_panel_preserves_fua_resolution_and_oecd_metadata(self) -> None:
        cfg = load_config()
        cities = pd.DataFrame(
            [
                {
                    "city_id": "darwin_au",
                    "city_name": "Darwin",
                    "country": "Australia",
                    "iso3": "AUS",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "city_id": "darwin_au",
                        "year": 2020,
                        "gdp_per_capita": 81234.5,
                        "gdp_total_ppp_observed": 12000000000.0,
                        "gdp_share_national_observed": 0.8,
                        "macro_observed_source": "oecd_fua_economy",
                        "macro_resolution_level": "fua_year",
                        "oecd_fua_code": "AUS16F",
                        "oecd_city_code": "AUS16C",
                        "oecd_fua_name": "Greater Darwin",
                        "oecd_city_name": "Greater Darwin",
                    }
                ]
            ).to_csv(tmp_path / "city_macro_observed.csv", index=False)
            bak_raw = gd.DATA_RAW
            gd.DATA_RAW = tmp_path
            try:
                out = gd.collect_city_macro_observed_panel(cities, cfg, use_cache=True)
            finally:
                gd.DATA_RAW = bak_raw

        observed_row = out[out["year"] == 2020].iloc[0]
        self.assertEqual(str(observed_row["macro_observed_source"]), "oecd_fua_economy")
        self.assertEqual(str(observed_row["macro_resolution_level"]), "fua_year")
        self.assertEqual(str(observed_row["oecd_fua_code"]), "AUS16F")
        self.assertEqual(str(observed_row["oecd_city_code"]), "AUS16C")
        self.assertEqual(str(observed_row["oecd_fua_name"]), "Greater Darwin")
        self.assertAlmostEqual(float(observed_row["gdp_per_capita"]), 81234.5, places=4)
        self.assertAlmostEqual(float(observed_row["gdp_total_ppp_observed"]), 12000000000.0, places=1)
        self.assertEqual(int(observed_row["city_macro_observed_flag"]), 1)

        unobserved_row = out[out["year"] != 2020].iloc[0]
        self.assertEqual(str(unobserved_row["macro_observed_source"]), "missing")
        self.assertEqual(str(unobserved_row["macro_resolution_level"]), "country_year")
        self.assertEqual(int(unobserved_row["city_macro_observed_flag"]), 0)

    def test_fit_single_target_imputes_sparse_viirs_feature(self) -> None:
        rows = []
        for city_idx in range(20):
            city_id = f"city_{city_idx:02d}"
            for year in range(2018, 2026):
                base = 40.0 + city_idx * 0.5 + (year - 2018) * 0.4
                rows.append(
                    {
                        "city_id": city_id,
                        "year": year,
                        "economic_vitality": base,
                        "livability": base + 3.0,
                        "innovation": base - 2.0,
                        "composite_index": base + 1.0,
                        "latitude": 10.0 + city_idx,
                        "longitude": 20.0 + city_idx,
                        "temperature_mean": 18.0 + 0.1 * city_idx,
                        "precipitation_sum": 100.0 + city_idx,
                        "climate_comfort": 0.6 + 0.01 * city_idx,
                        "amenity_ratio": 0.30 + 0.001 * city_idx,
                        "commerce_ratio": 0.25 + 0.001 * city_idx,
                        "transport_intensity": 0.10 + 0.001 * city_idx,
                        "poi_diversity": 0.80 + 0.002 * city_idx,
                        "observed_activity_signal": 0.4 + 0.01 * city_idx,
                        "observed_mobility_signal": 0.5 + 0.01 * city_idx,
                        "observed_dynamic_signal": 0.3 + 0.01 * city_idx,
                        "observed_livability_signal": 0.6 + 0.01 * city_idx,
                        "observed_innovation_signal": 0.2 + 0.01 * city_idx,
                        "observed_sentiment_signal": 0.1 + 0.01 * city_idx,
                        "observed_physical_stress_signal": 0.05 + 0.001 * city_idx,
                        "road_access_score": 50.0 + city_idx,
                        "road_tier_code": 2.0,
                        "road_length_km_total": 100.0 + city_idx,
                        "arterial_share": 0.30,
                        "intersection_density": 0.15 + 0.001 * city_idx,
                        "road_arterial_growth_proxy": 0.02,
                        "road_local_growth_proxy": 0.01,
                        "road_growth_intensity": 0.03,
                        "has_poi_observation": 1.0,
                        "poi_backcast_scale": 1.0,
                        "viirs_ntl_mean": 10.0 + city_idx,
                        "viirs_log_mean": 2.0 + 0.01 * city_idx,
                        "viirs_ntl_p90": 15.0 + city_idx,
                        "viirs_intra_year_recovery": 0.1,
                        "viirs_intra_year_decline": 0.05,
                        "viirs_recent_drop": 0.02,
                        "viirs_physical_continuity": 0.8,
                        "viirs_physical_stress": 0.2,
                        "viirs_ntl_yoy": np.nan if year == 2018 else 0.5,
                        "viirs_lit_area_km2": 30.0 + city_idx,
                        "has_viirs_observation": 1.0,
                        "viirs_year_coverage_share": 1.0,
                        "ghsl_built_surface_km2": 5.0 + city_idx,
                        "ghsl_built_volume_m3": 1000.0 + 10.0 * city_idx,
                        "ghsl_built_density": 200.0 + city_idx,
                        "ghsl_built_surface_yoy": 0.1,
                        "ghsl_built_volume_yoy": 0.2,
                        "ghsl_built_contraction": 0.0,
                        "osm_hist_road_length_m": 10000.0 + city_idx,
                        "osm_hist_building_count": 500.0 + city_idx,
                        "osm_hist_poi_count": 120.0 + city_idx,
                        "osm_hist_poi_food_count": 20.0 + city_idx,
                        "osm_hist_poi_retail_count": 15.0 + city_idx,
                        "osm_hist_poi_nightlife_count": 5.0 + city_idx,
                        "osm_hist_road_yoy": 0.03,
                        "osm_hist_building_yoy": 0.02,
                        "osm_hist_poi_yoy": 0.04,
                        "osm_hist_poi_food_yoy": 0.03,
                        "osm_hist_poi_retail_yoy": 0.03,
                        "osm_hist_poi_nightlife_yoy": 0.02,
                        "policy_event_count_iso_year": 1.0,
                        "policy_event_type_count_iso_year": 1.0,
                        "policy_event_new_count_iso_year": 1.0,
                        "policy_intensity_sum_iso_year": 0.5,
                        "policy_intensity_mean_iso_year": 0.5,
                        "policy_event_count_iso_year_yoy": 0.0,
                        "policy_intensity_sum_iso_year_yoy": 0.0,
                        "policy_news_proxy_score": 0.4,
                        "social_sentiment_score": 0.1,
                        "social_sentiment_volatility": 0.2,
                        "social_sentiment_positive_share": 0.6,
                        "social_sentiment_negative_share": 0.2,
                        "social_sentiment_buzz": 0.4,
                        "social_sentiment_delta_1": 0.0,
                        "has_social_observation": 1.0,
                    }
                )

        panel = pd.DataFrame(rows).sort_values(["city_id", "year"]).copy()
        for target in modeling.TARGETS:
            panel[f"{target}_t1"] = panel.groupby("city_id")[target].shift(-1)
        train, test = modeling._train_test_split_by_year(panel)

        metrics, model, best_name, fi, pred = modeling._fit_single_target(
            train,
            test,
            "composite_index",
            feature_cols=modeling._resolve_feature_columns(panel),
        )

        self.assertIn(best_name, metrics)
        self.assertFalse(pred["pred_best"].isna().any())
        self.assertIn("viirs_ntl_yoy", fi["feature"].tolist())

    def test_city_macro_disaggregation_uses_viirs_ntl_share_only(self) -> None:
        panel = pd.DataFrame(
            {
                "city_id": ["city_a", "city_b"],
                "iso3": ["AAA", "AAA"],
                "year": [2020, 2020],
                "gdp_per_capita": [1000.0, 1000.0],
                "population": [100.0, 100.0],
                "unemployment": [5.0, 5.0],
                "internet_users": [60.0, 60.0],
                "capital_formation": [25.0, 25.0],
                "inflation": [3.0, 3.0],
                "viirs_ntl_sum": [30.0, 70.0],
                "viirs_ntl_mean": [3.0, 7.0],
                "viirs_lit_area_km2": [10.0, 10.0],
            }
        )
        out, meta = gd._apply_city_macro_disaggregation(panel)
        self.assertEqual(meta.get("source"), "ntl_share_only_v1")
        self.assertTrue(np.allclose(out["gdp_per_capita"].to_numpy(dtype=float), np.array([1000.0, 1000.0])))
        self.assertTrue(np.allclose(out["gdp_disaggregation_weight_ntl_share"].to_numpy(dtype=float), np.array([0.3, 0.7])))
        self.assertTrue(np.allclose(out["gdp_disaggregated_by_ntl"].to_numpy(dtype=float), np.array([30000.0, 70000.0])))

    def test_econometrics_blocks_composite_index_as_causal_outcome(self) -> None:
        with self.assertRaises(ValueError):
            run_did_two_way_fe(pd.DataFrame())

    def test_modeling_uses_fixed_out_of_time_split(self) -> None:
        panel = pd.DataFrame(
            {
                "city_id": ["a"] * 6 + ["b"] * 6,
                "year": [2020, 2021, 2022, 2023, 2024, 2025] * 2,
            }
        )
        train, test = modeling._train_test_split_by_year(panel)
        self.assertLessEqual(int(train["year"].max()), 2022)
        self.assertGreaterEqual(int(test["year"].min()), 2023)

    def test_economic_vitality_feature_filter_drops_viirs_and_gdp(self) -> None:
        cols = [
            "temperature_mean",
            "precipitation_sum",
            "viirs_ntl_mean",
            "viirs_recent_drop",
            "gdp_growth",
            "knowledge_capital_raw",
        ]
        safe = modeling._leakage_safe_feature_columns("economic_vitality", cols)
        self.assertIn("temperature_mean", safe)
        self.assertIn("knowledge_capital_raw", safe)
        self.assertNotIn("viirs_ntl_mean", safe)
        self.assertNotIn("viirs_recent_drop", safe)
        self.assertNotIn("gdp_growth", safe)


if __name__ == "__main__":
    unittest.main()
