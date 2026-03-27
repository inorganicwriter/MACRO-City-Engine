from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

TEST_OUTPUT_ROOT = Path(tempfile.gettempdir()) / "macro_city_engine_test_artifacts"
os.environ["URBAN_PULSE_OUTPUT_ROOT"] = str(TEST_OUTPUT_ROOT)

from src.config import load_config
from src.city_catalog import load_city_catalog
from src.benchmark_eval import run_benchmark_suite
from src.causal_st import run_causal_st_analysis, run_causal_st_experiment_matrix
from src.econometrics import (
    run_did_heterogeneity,
    run_matched_did_with_trend,
    run_not_yet_treated_did,
    run_staggered_did,
    run_did_two_way_fe,
    run_twfe_cluster_bootstrap,
    run_twfe_city_permutation,
    run_twfe_wild_cluster_bootstrap,
    run_twfe_lead_placebo,
    run_dml_did,
    run_dr_did,
    run_matrix_completion_counterfactual,
)
from src.data_crawler import crawl_global_real_sources
from src.dynamic_causal_envelope import run_dynamic_causal_envelope_suite
from src.dynamic_method_core import run_dynamic_method_core_suite
from src.pulse_dynamics import run_pulse_dynamics_suite
from src.pulse_nowcast import run_pulse_nowcast_suite
from src.provenance import audit_and_filter_objective_sources, build_global_coverage_report
from src.pulse_state import estimate_pulse_states
from src.pulse_ai import run_pulse_ai_engine
from src.realtime_monitor import generate_realtime_monitor_snapshot
from src.experiment_enhancements import run_experiment_enhancements
from src.exogenous_shock_design import run_exogenous_shock_suite
from src.exogenous_shock_heterogeneity import run_exogenous_shock_heterogeneity_suite
from src.identification_plus import run_identification_plus_suite
from src.submission_extensions import run_submission_extensions
from src.top_tier_reinforcement import run_top_tier_reinforcement_suite
from src.utils import DATA_OUTPUTS, DATA_PROCESSED, minmax_scale, point_line_distance_km


class TestCore(unittest.TestCase):
    def test_config_load(self) -> None:
        cfg = load_config()
        self.assertGreater(cfg.max_cities_default, 0)
        self.assertLessEqual(cfg.time_range.start_year, cfg.time_range.end_year)

    def test_minmax_scale(self) -> None:
        arr = np.array([2.0, 4.0, 6.0])
        scaled = minmax_scale(arr)
        self.assertTrue(np.isclose(float(scaled.min()), 0.0))
        self.assertTrue(np.isclose(float(scaled.max()), 1.0))

    def test_point_line_distance(self) -> None:
        point = (0.0, 0.0)
        a = (0.0, 1.0)
        b = (0.0, 2.0)
        dist = point_line_distance_km(point, a, b)
        self.assertGreater(dist, 100.0)

    def test_did_runs(self) -> None:
        rows = []
        for city in ["a", "b"]:
            treated = 1 if city == "a" else 0
            for year in [2018, 2019, 2020, 2021]:
                post = 1 if year >= 2020 else 0
                did = treated * post
                y = 50 + 2.0 * did + (year - 2018) * 0.3 + (1 if city == "a" else -1)
                rows.append(
                    {
                        "city_id": city,
                        "year": year,
                        "composite_index": y,
                        "did_treatment": did,
                        "log_gdp_pc": 10.0 + 0.01 * year,
                        "internet_users": 50 + 0.2 * year,
                        "unemployment": 5.0,
                    }
                )
        panel = pd.DataFrame(rows)
        result = run_did_two_way_fe(panel)
        self.assertIn("coef", result)
        self.assertTrue(np.isfinite(result["coef"]))
        self.assertIn("stderr_type", result)
        self.assertIn("n_clusters", result)

        panel2 = panel.copy()
        panel2["treated_city"] = np.where(panel2["city_id"] == "a", 1, 0)
        panel2["post_policy"] = (panel2["year"] >= 2020).astype(int)
        panel2["did_treatment"] = panel2["treated_city"] * panel2["post_policy"]
        bs = run_twfe_cluster_bootstrap(panel2, draws=30, random_state=7)
        self.assertTrue((bs.get("status") == "ok") or (bs.get("status") == "skipped"))
        perm = run_twfe_city_permutation(panel2, draws=40, random_state=7)
        self.assertTrue((perm.get("status") == "ok") or (perm.get("status") == "skipped"))
        wild = run_twfe_wild_cluster_bootstrap(panel2, draws=50, random_state=7)
        self.assertTrue((wild.get("status") == "ok") or (wild.get("status") == "skipped"))
        lead = run_twfe_lead_placebo(panel2, lead_years=[1, 2], track_label="ut")
        self.assertTrue((lead.get("status") == "ok") or (lead.get("status") == "skipped"))

    def test_did_heterogeneity_runs(self) -> None:
        rows = []
        cities = [("a", "Asia"), ("b", "Asia"), ("c", "Europe"), ("d", "Europe")]
        for city, continent in cities:
            treated = 1 if city in {"a", "c"} else 0
            for year in range(2018, 2023):
                post = 1 if year >= 2020 else 0
                did = treated * post
                rows.append(
                    {
                        "city_id": city,
                        "city_name": city,
                        "continent": continent,
                        "year": year,
                        "treated_city": treated,
                        "post_policy": post,
                        "did_treatment": did,
                        "composite_index": 55 + 1.4 * did + 0.2 * (year - 2018),
                        "log_gdp_pc": 9.0 + 0.03 * year + (0.4 if continent == "Europe" else 0.0),
                        "internet_users": 60 + 0.1 * year + (2 if treated else -1),
                        "unemployment": 5.5 - 0.05 * (year - 2018) + (0.3 if continent == "Europe" else 0.0),
                        "gdp_per_capita": 10000 + 2000 * (1 if continent == "Europe" else 0) + 300 * (year - 2018),
                        "population": 2_000_000 + 20_000 * (year - 2018),
                    }
                )
        panel = pd.DataFrame(rows)
        out = run_did_heterogeneity(panel)
        self.assertIn("by_continent", out)
        self.assertIn("Asia", out["by_continent"])

    def test_dml_and_matrix_completion_runs(self) -> None:
        rows = []
        cities = [("a", "Asia"), ("b", "Asia"), ("c", "Europe"), ("d", "Europe"), ("e", "Europe"), ("f", "Asia")]
        for city, continent in cities:
            treated = 1 if city in {"a", "c"} else 0
            for year in range(2015, 2026):
                post = 1 if year >= 2020 else 0
                did = treated * post
                y = 50 + 1.2 * did + 0.15 * (year - 2015) + (1.0 if continent == "Europe" else -0.5)
                rows.append(
                    {
                        "city_id": city,
                        "city_name": city,
                        "continent": continent,
                        "year": year,
                        "treated_city": treated,
                        "post_policy": post,
                        "did_treatment": did,
                        "composite_index": y,
                        "log_gdp_pc": 9.0 + 0.02 * year + (0.3 if continent == "Europe" else 0.0),
                        "internet_users": 55 + 0.2 * year + (1 if treated else 0),
                        "unemployment": 6.0 - 0.03 * (year - 2015) + (0.2 if continent == "Europe" else 0.0),
                        "gdp_per_capita": 12000 + 3000 * (1 if continent == "Europe" else 0) + 200 * (year - 2015),
                        "population": 2_000_000 + 15_000 * (year - 2015),
                        "log_population": np.log(2_000_000 + 15_000 * (year - 2015)),
                        "capital_formation": 24 + 0.1 * (year - 2015),
                        "inflation": 3.0 + 0.05 * (year - 2015),
                    }
                )
        panel = pd.DataFrame(rows)

        dml = run_dml_did(panel)
        self.assertTrue(("coef" in dml) or (dml.get("status") == "skipped"))

        dr = run_dr_did(panel)
        self.assertTrue(("coef" in dr) or (dr.get("status") == "skipped"))

        mc = run_matrix_completion_counterfactual(panel, treatment_year=2020, placebo_count=3)
        self.assertTrue(("att_post" in mc) or (mc.get("status") == "skipped"))

    def test_dynamic_causal_envelope_runs(self) -> None:
        event = pd.DataFrame(
            [
                {"variant": "v1", "rel_year": -2, "coef": -0.05, "stderr": 0.02, "t_value": -2.5},
                {"variant": "v2", "rel_year": -2, "coef": -0.02, "stderr": 0.02, "t_value": -1.0},
                {"variant": "v1", "rel_year": 0, "coef": 0.08, "stderr": 0.03, "t_value": 2.7},
                {"variant": "v2", "rel_year": 0, "coef": 0.04, "stderr": 0.03, "t_value": 1.3},
            ]
        )
        event.to_csv(DATA_OUTPUTS / "econometric_source_event_study_points.csv", index=False)
        pd.DataFrame(
            [
                {"variant": "v1", "resilience_score_0_100": 72},
                {"variant": "v2", "resilience_score_0_100": 48},
            ]
        ).to_csv(DATA_OUTPUTS / "idplus_identification_stress_index.csv", index=False)
        pd.DataFrame(
            [
                {"event_type": "enter_deep_cooling", "event_time": -1, "n_obs": 10, "mean_rel_growth_adj": 0.0, "mean_rel_risk_adj": 0.0},
                {"event_type": "enter_deep_cooling", "event_time": 0, "n_obs": 10, "mean_rel_growth_adj": -0.4, "mean_rel_risk_adj": 0.6},
                {"event_type": "enter_deep_cooling", "event_time": 1, "n_obs": 10, "mean_rel_growth_adj": -0.1, "mean_rel_risk_adj": 0.2},
            ]
        ).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_state_event_study.csv", index=False)
        pd.DataFrame(
            [
                {
                    "city_id": "a",
                    "city_name": "A",
                    "country": "X",
                    "continent": "Asia",
                    "year": 2023,
                    "dynamic_pulse_index": 52.0,
                    "dynamic_pulse_delta_1y": 0.2,
                    "dynamic_pulse_trend_3y": 0.5,
                    "pulse_accel_velocity": 0.3,
                    "pulse_risk_velocity": -0.2,
                },
                {
                    "city_id": "a",
                    "city_name": "A",
                    "country": "X",
                    "continent": "Asia",
                    "year": 2024,
                    "dynamic_pulse_index": 53.0,
                    "dynamic_pulse_delta_1y": 1.0,
                    "dynamic_pulse_trend_3y": 0.7,
                    "pulse_accel_velocity": 0.4,
                    "pulse_risk_velocity": -0.1,
                },
                {
                    "city_id": "a",
                    "city_name": "A",
                    "country": "X",
                    "continent": "Asia",
                    "year": 2025,
                    "dynamic_pulse_index": 55.0,
                    "dynamic_pulse_delta_1y": 2.0,
                    "dynamic_pulse_trend_3y": 1.2,
                    "pulse_accel_velocity": 0.5,
                    "pulse_risk_velocity": -0.1,
                },
                {
                    "city_id": "b",
                    "city_name": "B",
                    "country": "Y",
                    "continent": "Africa",
                    "year": 2023,
                    "dynamic_pulse_index": 45.0,
                    "dynamic_pulse_delta_1y": -0.4,
                    "dynamic_pulse_trend_3y": -0.6,
                    "pulse_accel_velocity": -0.3,
                    "pulse_risk_velocity": 0.4,
                },
                {
                    "city_id": "b",
                    "city_name": "B",
                    "country": "Y",
                    "continent": "Africa",
                    "year": 2024,
                    "dynamic_pulse_index": 44.0,
                    "dynamic_pulse_delta_1y": -1.0,
                    "dynamic_pulse_trend_3y": -0.8,
                    "pulse_accel_velocity": -0.4,
                    "pulse_risk_velocity": 0.6,
                },
                {
                    "city_id": "b",
                    "city_name": "B",
                    "country": "Y",
                    "continent": "Africa",
                    "year": 2025,
                    "dynamic_pulse_index": 42.0,
                    "dynamic_pulse_delta_1y": -2.0,
                    "dynamic_pulse_trend_3y": -1.2,
                    "pulse_accel_velocity": -0.5,
                    "pulse_risk_velocity": 0.8,
                },
            ]
        ).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_index_series.csv", index=False)
        pd.DataFrame(
            [
                {"city_id": "a", "stall_risk_score": 32.0, "dynamic_hazard_fused_score": 35.0, "trajectory_regime": "stable_mature", "kinetic_state": "cooling"},
                {"city_id": "b", "stall_risk_score": 78.0, "dynamic_hazard_fused_score": 80.0, "trajectory_regime": "structural_decline", "kinetic_state": "overheating"},
            ]
        ).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_hazard_latest.csv", index=False)
        pd.DataFrame(
            [
                {"city_id": "a", "acceleration_score": 68.0, "stall_risk_score": 32.0},
                {"city_id": "b", "acceleration_score": 28.0, "stall_risk_score": 78.0},
            ]
        ).to_csv(DATA_OUTPUTS / "pulse_ai_city_latest.csv", index=False)

        out = run_dynamic_causal_envelope_suite()
        self.assertEqual(out.get("status"), "ok")
        self.assertTrue((DATA_OUTPUTS / "dynamic_causal_envelope_summary.json").exists())
        self.assertTrue((DATA_OUTPUTS / "dynamic_causal_envelope_city_scores.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "dynamic_causal_envelope_event_bootstrap.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "dynamic_causal_envelope_continent_stability.csv").exists())
        self.assertIn("event_post_ci_above_zero_share", out)
        self.assertIn("mean_continent_stability_score", out)

    def test_pulse_dynamics_suite_runs(self) -> None:
        rows = []
        cities = [
            ("a", "A", "X", "Asia", 0.10),
            ("b", "B", "Y", "Africa", 0.35),
            ("c", "C", "Z", "Europe", 0.20),
            ("d", "D", "Q", "South America", 0.42),
        ]
        years = list(range(2018, 2026))
        for city_id, city_name, country, continent, phase in cities:
            for idx, year in enumerate(years):
                base = 45 + 14 * np.sin((idx + 1) * 0.8 + phase) + (8 if continent in {"Africa", "South America"} else 0)
                risk = float(np.clip(base + (2.0 if (idx % 3 == 0) else -1.5), 5, 95))
                accel = float(np.clip(72 - 0.5 * risk + 10 * np.cos((idx + 2) * 0.9 + phase), 5, 95))
                stall_next = float(1.0 if (risk >= 58 and accel <= 52) else 0.0)
                pred = float(np.clip(0.01 + risk / 120.0 + (0.08 if stall_next > 0 else -0.02), 0.01, 0.99))
                rows.append(
                    {
                        "city_id": city_id,
                        "city_name": city_name,
                        "country": country,
                        "continent": continent,
                        "year": year,
                        "stall_risk_score": risk,
                        "acceleration_score": accel,
                        "stall_next": stall_next,
                        "dynamic_hazard_fused_probability": pred,
                        "stall_probability": float(np.clip(pred - 0.04, 0.01, 0.99)),
                    }
                )
        pd.DataFrame(rows).to_csv(DATA_OUTPUTS / "pulse_ai_scores.csv", index=False)

        out = run_pulse_dynamics_suite()
        self.assertEqual(out.get("status"), "ok")
        self.assertTrue((DATA_OUTPUTS / "pulse_dynamics_summary.json").exists())
        self.assertTrue((DATA_OUTPUTS / "pulse_dynamics_transition_tensor.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "pulse_dynamics_spell_hazard.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "pulse_dynamics_resilience_halflife.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "pulse_dynamics_warning_horizon.csv").exists())
        self.assertIn("global_warning_auc_h2", out)
        self.assertIn("global_stall_exit_half_life", out)

    def test_pulse_nowcast_suite_runs(self) -> None:
        rows = []
        for cont, base, amp in [
            ("Africa", 55, 7),
            ("Asia", 62, 6),
            ("Europe", 66, 5),
            ("North America", 64, 5),
            ("South America", 58, 6),
            ("Oceania", 63, 4),
        ]:
            for i, year in enumerate(range(2015, 2026)):
                val = base + amp * np.sin((i + 1) * 0.7) + 0.35 * i
                rows.append(
                    {
                        "continent": cont,
                        "year": year,
                        "dynamic_pulse_index_mean": float(np.clip(val, 20, 95)),
                        "dynamic_pulse_index_p75": float(np.clip(val + 4.0, 20, 99)),
                        "dynamic_pulse_delta_1y_mean": 0.0,
                        "city_count": 40 + (i % 5),
                    }
                )
        pd.DataFrame(rows).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_index_continent_year.csv", index=False)

        out = run_pulse_nowcast_suite()
        self.assertEqual(out.get("status"), "ok")
        self.assertTrue((DATA_OUTPUTS / "pulse_nowcast_summary.json").exists())
        self.assertTrue((DATA_OUTPUTS / "pulse_nowcast_continent_latest.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "pulse_nowcast_continent_history.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "pulse_nowcast_backtest_metrics.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "pulse_nowcast_global.csv").exists())
        self.assertIn("global_backtest_directional_accuracy", out)

    def test_dynamic_method_core_suite_runs(self) -> None:
        rows = []
        continents = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
        for ci, cont in enumerate(continents):
            for city_idx in range(2):
                city_id = f"{cont[:2].lower()}_{city_idx}"
                city_name = f"{cont}_City_{city_idx}"
                base = 48.0 + (ci * 3.0) + (city_idx * 1.2)
                amp = 5.0 + (ci % 3)
                for year in range(2015, 2026):
                    t = float(year - 2015)
                    idx = base + amp * np.sin(0.55 * t + 0.25 * ci) + 0.42 * t + 0.5 * city_idx
                    prev = base + amp * np.sin(0.55 * (t - 1.0) + 0.25 * ci) + 0.42 * (t - 1.0) + 0.5 * city_idx
                    delta = idx - prev if year > 2015 else 0.0
                    rows.append(
                        {
                            "city_id": city_id,
                            "city_name": city_name,
                            "country": f"C_{cont[:2]}",
                            "continent": cont,
                            "year": year,
                            "dynamic_pulse_index": float(np.clip(idx, 10, 95)),
                            "dynamic_pulse_delta_1y": float(delta),
                            "dynamic_pulse_trend_3y": float(0.9 * delta),
                            "dynamic_pulse_state": "mid_transition",
                            "pulse_accel_velocity": float(0.35 * delta),
                            "pulse_risk_velocity": float(-0.25 * delta),
                        }
                    )
        pd.DataFrame(rows).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_index_series.csv", index=False)

        out = run_dynamic_method_core_suite()
        self.assertEqual(out.get("status"), "ok")
        self.assertTrue((DATA_OUTPUTS / "dynamic_method_core_summary.json").exists())
        self.assertTrue((DATA_OUTPUTS / "dynamic_method_core_predictions.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "dynamic_method_core_metrics.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "dynamic_method_core_significance.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "dynamic_method_core_ablation.csv").exists())

    def test_city_catalog_balanced_sampling(self) -> None:
        sampled = load_city_catalog(max_cities=60)
        self.assertEqual(len(sampled), 60)
        self.assertEqual(sampled["city_id"].nunique(), 60)
        self.assertEqual(sampled["continent"].nunique(), 6)
        self.assertIn("Africa", set(sampled["continent"]))
        self.assertIn("South America", set(sampled["continent"]))
        self.assertGreaterEqual(sampled["country"].nunique(), 30)

    def test_provenance_filter_strict(self) -> None:
        panel = pd.DataFrame(
            [
                {
                    "city_id": "a",
                    "city_name": "A",
                    "country": "X",
                    "continent": "Asia",
                    "year": 2020,
                    "macro_source": "world_bank",
                    "weather_source": "open-meteo",
                    "poi_source": "osm",
                },
                {
                    "city_id": "a",
                    "city_name": "A",
                    "country": "X",
                    "continent": "Asia",
                    "year": 2021,
                    "macro_source": "world_bank",
                    "weather_source": "open-meteo",
                    "poi_source": "osm",
                },
                {
                    "city_id": "b",
                    "city_name": "B",
                    "country": "Y",
                    "continent": "Europe",
                    "year": 2020,
                    "macro_source": "world_bank",
                    "weather_source": "nasa-power",
                    "poi_source": "missing",
                },
                {
                    "city_id": "b",
                    "city_name": "B",
                    "country": "Y",
                    "continent": "Europe",
                    "year": 2021,
                    "macro_source": "world_bank",
                    "weather_source": "nasa-power",
                    "poi_source": "osm",
                },
            ]
        )
        filtered, audit = audit_and_filter_objective_sources(panel, strict_mode=True)
        self.assertEqual(filtered["city_id"].nunique(), 1)
        self.assertEqual(filtered["city_id"].iloc[0], "a")
        self.assertGreater(audit["dropped"]["rows"], 0)

    def test_provenance_filter_verified_toggle(self) -> None:
        panel = pd.DataFrame(
            [
                {
                    "city_id": "a",
                    "city_name": "A",
                    "country": "X",
                    "continent": "Asia",
                    "year": 2020,
                    "macro_source": "world_bank",
                    "weather_source": "open-meteo",
                    "poi_source": "osm",
                },
                {
                    "city_id": "a",
                    "city_name": "A",
                    "country": "X",
                    "continent": "Asia",
                    "year": 2021,
                    "macro_source": "world_bank",
                    "weather_source": "open-meteo",
                    "poi_source": "osm",
                },
                {
                    "city_id": "b",
                    "city_name": "B",
                    "country": "Y",
                    "continent": "Africa",
                    "year": 2020,
                    "macro_source": "world_bank",
                    "weather_source": "nasa-power",
                    "poi_source": "imputed_from_osm_pool",
                },
                {
                    "city_id": "b",
                    "city_name": "B",
                    "country": "Y",
                    "continent": "Africa",
                    "year": 2021,
                    "macro_source": "world_bank",
                    "weather_source": "nasa-power",
                    "poi_source": "imputed_from_osm_pool",
                },
            ]
        )
        filtered_verified, audit_verified = audit_and_filter_objective_sources(
            panel,
            strict_mode=True,
            enforce_verified=True,
        )
        self.assertEqual(set(filtered_verified["city_id"]), {"a"})
        self.assertEqual(audit_verified.get("filter_basis"), "verified_city_complete")

        filtered_objective, audit_objective = audit_and_filter_objective_sources(
            panel,
            strict_mode=True,
            enforce_verified=False,
        )
        self.assertEqual(set(filtered_objective["city_id"]), {"a", "b"})
        self.assertEqual(audit_objective.get("filter_basis"), "objective_city_complete")

    def test_coverage_report_runs(self) -> None:
        panel = pd.DataFrame(
            [
                {
                    "city_id": "a",
                    "city_name": "A",
                    "country": "X",
                    "continent": "Asia",
                    "year": 2020,
                    "macro_source": "world_bank",
                    "weather_source": "open-meteo",
                    "poi_source": "osm",
                },
                {
                    "city_id": "a",
                    "city_name": "A",
                    "country": "X",
                    "continent": "Asia",
                    "year": 2021,
                    "macro_source": "world_bank",
                    "weather_source": "open-meteo",
                    "poi_source": "osm",
                },
                {
                    "city_id": "b",
                    "city_name": "B",
                    "country": "Y",
                    "continent": "Africa",
                    "year": 2020,
                    "macro_source": "world_bank",
                    "weather_source": "nasa-power",
                    "poi_source": "imputed_from_osm_pool",
                },
                {
                    "city_id": "b",
                    "city_name": "B",
                    "country": "Y",
                    "continent": "Africa",
                    "year": 2021,
                    "macro_source": "world_bank",
                    "weather_source": "nasa-power",
                    "poi_source": "imputed_from_osm_pool",
                },
            ]
        )
        out = build_global_coverage_report(panel, strict_mode=True)
        self.assertEqual(out.get("status"), "ok")
        self.assertEqual(out.get("n_countries"), 2)
        self.assertEqual(out.get("n_continents"), 2)

    def test_realtime_monitor_snapshot_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact_dir = root / "data" / "outputs" / "realtime"
            processed_dir = root / "data" / "processed"
            outputs_dir = root / "data" / "outputs"
            reports_dir = root / "reports"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            processed_dir.mkdir(parents=True, exist_ok=True)
            outputs_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)

            city_points = pd.DataFrame(
                [
                    {
                        "city_id": "a",
                        "city_name": "A",
                        "country": "X",
                        "continent": "Asia",
                        "year": y,
                        "latitude": 0.0,
                        "longitude": 0.0,
                        "economic_vitality": 50 + y - 2020,
                        "livability": 52 + y - 2020,
                        "innovation": 54 + y - 2020,
                        "composite_index": 51 + y - 2020,
                        "treated_city": 1,
                    }
                    for y in [2023, 2024, 2025]
                ]
                + [
                    {
                        "city_id": "b",
                        "city_name": "B",
                        "country": "Y",
                        "continent": "Europe",
                        "year": y,
                        "latitude": 1.0,
                        "longitude": 1.0,
                        "economic_vitality": 48 + 0.5 * (y - 2023),
                        "livability": 49 + 0.2 * (y - 2023),
                        "innovation": 47 + 0.3 * (y - 2023),
                        "composite_index": 48 + 0.4 * (y - 2023),
                        "treated_city": 0,
                    }
                    for y in [2023, 2024, 2025]
                ]
            )
            city_points.to_csv(outputs_dir / "city_points.csv", index=False)

            pulse = pd.DataFrame(
                [
                    {
                        "city_id": "a",
                        "acceleration_score": 70.0,
                        "stall_risk_score": 35.0,
                        "stall_risk_interval_width": 4.0,
                        "accel_shift_1y": 6.0,
                        "forecast_risk_delta_h1": -8.0,
                    },
                    {
                        "city_id": "b",
                        "acceleration_score": 35.0,
                        "stall_risk_score": 85.0,
                        "stall_risk_interval_width": 6.0,
                        "accel_shift_1y": -4.0,
                        "forecast_risk_delta_h1": 10.0,
                    },
                ]
            )
            pulse.to_csv(outputs_dir / "pulse_ai_city_latest.csv", index=False)

            pd.DataFrame(
                [
                    {"city_id": "a", "verified_ratio": 1.0, "objective_ratio": 1.0, "is_verified_complete_city": 1},
                    {"city_id": "b", "verified_ratio": 0.8, "objective_ratio": 1.0, "is_verified_complete_city": 0},
                ]
            ).to_csv(processed_dir / "source_audit_city.csv", index=False)

            with (processed_dir / "source_audit_summary.json").open("w", encoding="utf-8") as f:
                json.dump({"enforce_verified": True, "verified_row_ratio": 0.9, "city_retention_ratio": 0.8}, f)
            with (reports_dir / "pipeline_summary.json").open("w", encoding="utf-8") as f:
                json.dump({"reliability_gate": {"status": "ok"}}, f)

            status = generate_realtime_monitor_snapshot(
                trigger="unit_test",
                artifact_dir=artifact_dir,
                data_processed_dir=processed_dir,
                data_outputs_dir=outputs_dir,
                reports_dir=reports_dir,
                write_versioned_snapshot=True,
            )
            self.assertEqual(status.get("status"), "ok")
            self.assertEqual(status.get("city_count"), 2)
            self.assertEqual(status.get("country_count"), 2)
            self.assertEqual(status.get("latest_data_year"), 2025)
            self.assertIn("sentinel_city_count", status)
            self.assertIn("sentinel_break_count", status)

            self.assertTrue((artifact_dir / "realtime_status.json").exists())
            self.assertTrue((artifact_dir / "realtime_country_monitor.csv").exists())
            self.assertTrue((artifact_dir / "realtime_alerts.csv").exists())
            self.assertTrue((artifact_dir / "realtime_sentinel.csv").exists())
            self.assertTrue((outputs_dir / "realtime_monitor_history.jsonl").exists())

    def test_pulse_state_runs(self) -> None:
        rows = []
        for city in ["a", "b", "c"]:
            for year in range(2017, 2025):
                rows.append(
                    {
                        "city_id": city,
                        "city_name": city,
                        "year": year,
                        "composite_index": 40 + 2.5 * (year - 2017) + (3 if city == "a" else -2 if city == "c" else 0),
                    }
                )
        panel = pd.DataFrame(rows)
        out = estimate_pulse_states(panel)
        self.assertEqual(out["n_cities"], 3)
        self.assertEqual(set(out["states"]), {
            "accelerating_expansion",
            "decelerating_expansion",
            "deepening_contraction",
            "bottoming_recovery",
        })

    def test_pulse_ai_runs(self) -> None:
        rows = []
        cities = [("a", "A", "X", "Asia"), ("b", "B", "Y", "Europe"), ("c", "C", "Z", "Africa"), ("d", "D", "W", "South America")]
        for idx, (cid, cname, country, continent) in enumerate(cities):
            for year in range(2015, 2026):
                base = 45 + 0.9 * (year - 2015) + 1.2 * idx
                y = base + (2.5 if (cid in {"a", "b"} and year >= 2020) else -0.8 if (cid == "d" and year >= 2021) else 0.0)
                rows.append(
                    {
                        "city_id": cid,
                        "city_name": cname,
                        "country": country,
                        "continent": continent,
                        "year": year,
                        "composite_index": y,
                        "economic_vitality": y * 0.96,
                        "livability": y * 0.91,
                        "innovation": y * 1.04,
                        "gdp_growth": 0.015 + 0.001 * (year - 2015),
                        "internet_users": 52 + 1.1 * (year - 2015),
                        "unemployment": 7.2 - 0.08 * (year - 2015) + 0.12 * idx,
                        "capital_formation": 21 + 0.11 * (year - 2015) + 0.25 * idx,
                        "inflation": 2.4 + 0.03 * (year - 2015),
                    }
                )
        panel = pd.DataFrame(rows)
        out = run_pulse_ai_engine(panel)
        self.assertIn("model_metrics", out)
        self.assertEqual(out["n_cities"], 4)
        self.assertIn("top_stall_risk_cities", out)
        self.assertIn("cross_continent_generalization", out)
        self.assertIn("shock_pulse_response", out)
        self.assertIn("model_variant_selection", out)
        self.assertIn("continent_calibration", out)
        self.assertIn("uncertainty_quantification", out)
        self.assertIn("dynamic_structure", out)
        self.assertIn("critical_transition", out["dynamic_structure"])
        self.assertIn("graph_diffusion", out["dynamic_structure"])
        self.assertIn("main_risk_fusion", out["dynamic_structure"])
        self.assertIn("state_event_effects", out["dynamic_structure"])
        self.assertIn("sync_network", out["dynamic_structure"])
        self.assertIn("policy_lab", out["dynamic_structure"])
        self.assertIn("policy_rl", out["dynamic_structure"])
        self.assertIn("pulse_index", out["dynamic_structure"])
        self.assertIn("phase_portrait", out["dynamic_structure"])
        policy_rl = out["dynamic_structure"]["policy_rl"]
        self.assertTrue(("status" in policy_rl) and (policy_rl["status"] in {"ok", "skipped"}))
        if policy_rl.get("status") == "ok":
            self.assertIn("selected_action_distribution", policy_rl)
            self.assertIn("selection_source_distribution", policy_rl)
            self.assertIn("offline_policy_evaluation", policy_rl)
        pulse_index = out["dynamic_structure"]["pulse_index"]
        self.assertTrue(("status" in pulse_index) and (pulse_index["status"] in {"ok", "skipped"}))
        fusion = out["dynamic_structure"]["main_risk_fusion"]
        self.assertTrue(("selected_alpha_graph" in fusion) or (fusion.get("status") == "skipped"))
        self.assertTrue(("selected_alpha_critical" in fusion) or (fusion.get("status") == "skipped"))
        self.assertTrue(("selection_policy" in fusion) or (fusion.get("status") == "skipped"))
        self.assertTrue(("selection_rank_score" in fusion) or (fusion.get("status") == "skipped"))
        self.assertTrue(("post_fusion_shrink" in fusion) or (fusion.get("status") == "skipped"))
        self.assertTrue(("post_fusion_calibration" in fusion) or (fusion.get("status") == "skipped"))
        self.assertTrue(("post_fusion_geo_regularization" in fusion) or (fusion.get("status") == "skipped"))
        self.assertTrue(("multi_objective_frontier" in fusion) or (fusion.get("status") == "skipped"))
        self.assertTrue(("state_gated_moe" in fusion) or (fusion.get("status") == "skipped"))
        self.assertTrue(("continent_generalization" in fusion) or (fusion.get("status") == "skipped"))
        self.assertTrue(("continent_robust_selection" in fusion) or (fusion.get("status") == "skipped"))
        self.assertTrue(("continent_adaptive_regularization" in fusion) or (fusion.get("status") == "skipped"))
        self.assertIn("multi_horizon_forecast", out)

    def test_causal_st_and_benchmark_run(self) -> None:
        rows = []
        cities = [f"c{i}" for i in range(20)]
        for i, city in enumerate(cities):
            cont = "Asia" if i < 5 else "Europe" if i < 10 else "North America" if i < 15 else "South America"
            treated = 1 if i % 4 == 0 else 0
            for year in range(2016, 2025):
                post = 1 if year >= 2020 else 0
                did = treated * post
                base = 50 + 0.8 * (year - 2016) + (i % 5)
                y = base + 1.2 * did
                rows.append(
                    {
                        "city_id": city,
                        "city_name": city,
                        "country": f"cty{i%7}",
                        "continent": cont,
                        "latitude": 10 + i * 0.5,
                        "longitude": 20 + i * 0.6,
                        "year": year,
                        "treated_city": treated,
                        "post_policy": post,
                        "did_treatment": did,
                        "composite_index": y,
                        "economic_vitality": y * 0.95,
                        "livability": y * 0.9,
                        "innovation": y * 1.05,
                        "log_gdp_pc": 9.0 + 0.01 * year,
                        "log_population": np.log(1_000_000 + 10_000 * (year - 2016)),
                        "gdp_growth": 0.02 + 0.001 * (year - 2016),
                        "internet_users": 55 + 0.8 * (year - 2016),
                        "unemployment": 6.0 - 0.05 * (year - 2016),
                        "capital_formation": 23 + 0.1 * (year - 2016),
                        "inflation": 3.0 + 0.02 * (year - 2016),
                        "temperature_mean": 16 + 0.1 * i,
                        "precipitation_sum": 700 + 5 * i,
                        "climate_comfort": 0.7,
                        "amenity_ratio": 0.4,
                        "commerce_ratio": 0.3,
                        "transport_intensity": 0.2,
                        "poi_total": 1200 + 10 * i,
                        "poi_diversity": 1.1,
                        "macro_source": "world_bank",
                        "weather_source": "open-meteo",
                        "poi_source": "osm",
                    }
                )
        panel = pd.DataFrame(rows)
        cst = run_causal_st_analysis(panel)
        self.assertTrue(("att_post" in cst) or (cst.get("status") == "skipped"))
        if "att_post" in cst:
            # Verify new cross-fitting SE fields are present
            self.assertIn("se_method", cst)
            self.assertIn("att_ci95", cst)
            self.assertEqual(len(cst["att_ci95"]), 2)
            # Sanity guard: t-value should not be unreasonably inflated
            # (the old code produced t≈9 due to in-sample pseudo-variance; bootstrap should be much lower)
            t_abs = abs(float(cst.get("t_value", 0.0)))
            self.assertLess(t_abs, 15.0, f"Causal-ST t={t_abs:.2f} looks unreasonably large — check cross-fitting.")

        ab = run_causal_st_experiment_matrix(panel)
        self.assertIn("variants", ab)

        bm = run_benchmark_suite(panel)
        self.assertTrue(("temporal_holdout" in bm) or (bm.get("status") == "skipped"))

        matched = run_matched_did_with_trend(panel, treatment_year=2020)
        self.assertTrue(("coef" in matched) or (matched.get("status") == "skipped"))

        staggered = run_staggered_did(panel, treatment_year=2020)
        self.assertTrue(("post_avg_att" in staggered) or (staggered.get("status") == "skipped"))

        nyt = run_not_yet_treated_did(panel, treatment_year=2020)
        self.assertTrue(("att_weighted" in nyt) or (nyt.get("status") == "skipped"))

    def test_experiment_enhancements_fast_mode(self) -> None:
        rows = []
        cities = [f"c{i}" for i in range(24)]
        continents = ["Asia", "Europe", "North America", "South America", "Africa", "Oceania"]
        for i, city in enumerate(cities):
            cont = continents[i % len(continents)]
            treated = 1 if i % 3 == 0 else 0
            for year in range(2015, 2026):
                post = 1 if year >= 2020 else 0
                did = treated * post
                base = 42 + 0.85 * (year - 2015) + (i % 5)
                y = base + 0.9 * did + (0.2 if cont in {"Asia", "Europe"} else -0.1)
                rows.append(
                    {
                        "city_id": city,
                        "city_name": city,
                        "country": f"cty{i%9}",
                        "continent": cont,
                        "latitude": 8 + i * 0.7,
                        "longitude": 16 + i * 0.8,
                        "year": year,
                        "treated_city": treated,
                        "post_policy": post,
                        "did_treatment": did,
                        "composite_index": y,
                        "economic_vitality": y * 0.97,
                        "livability": y * 0.92,
                        "innovation": y * 1.04,
                        "log_gdp_pc": 8.6 + 0.01 * year,
                        "log_population": np.log(1_200_000 + 12000 * (year - 2015)),
                        "gdp_growth": 0.018 + 0.001 * (year - 2015),
                        "internet_users": 48 + 1.2 * (year - 2015),
                        "unemployment": 7.0 - 0.04 * (year - 2015),
                        "capital_formation": 21 + 0.12 * (year - 2015),
                        "inflation": 2.8 + 0.03 * (year - 2015),
                        "temperature_mean": 15 + 0.2 * (i % 5),
                        "precipitation_sum": 640 + 7 * (i % 6),
                        "climate_comfort": 0.72,
                        "amenity_ratio": 0.33,
                        "commerce_ratio": 0.31,
                        "transport_intensity": 0.21,
                        "poi_total": 950 + 18 * i,
                        "poi_diversity": 1.05,
                        "macro_source": "world_bank",
                        "weather_source": "open-meteo",
                        "poi_source": "osm",
                    }
                )
        panel = pd.DataFrame(rows)
        run_pulse_ai_engine(panel)
        out = run_experiment_enhancements(panel, fast_mode=True)
        self.assertIn("pulse_uncertainty_calibration", out)
        self.assertIn("pulse_group_fairness", out)
        self.assertIn("pulse_decision_curve", out)
        self.assertIn("did_permutation", out)
        self.assertIn("did_specification_curve", out)

    def test_top_tier_reinforcement_suite_runs(self) -> None:
        DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {
                    "candidate": "convex_blend",
                    "alpha_dynamic": 0.3,
                    "alpha_graph": 0.2,
                    "alpha_critical": 0.1,
                    "alpha_base": 0.4,
                    "eval_roc_auc": 0.79,
                    "eval_average_precision": 0.71,
                    "eval_brier": 0.23,
                    "temporal_mean_gain_vs_base": 0.02,
                    "temporal_share_positive_gain": 0.9,
                },
                {
                    "candidate": "convex_blend",
                    "alpha_dynamic": 0.2,
                    "alpha_graph": 0.1,
                    "alpha_critical": 0.1,
                    "alpha_base": 0.6,
                    "eval_roc_auc": 0.77,
                    "eval_average_precision": 0.68,
                    "eval_brier": 0.24,
                    "temporal_mean_gain_vs_base": 0.01,
                    "temporal_share_positive_gain": 0.7,
                },
            ]
        ).to_csv(DATA_OUTPUTS / "pulse_ai_dynamic_main_fusion_pareto.csv", index=False)

        pd.DataFrame(
            [
                {
                    "variant": "main_design",
                    "status": "ok",
                    "design_variant": "intense_external_peak",
                    "did_coef": 0.05,
                    "did_stderr": 0.02,
                    "did_t_value": 2.5,
                    "did_p_value": 0.012,
                    "bootstrap_p_value": 0.05,
                    "permutation_p_value": 0.07,
                    "wild_bootstrap_p_value": 0.08,
                    "lead_placebo_share_p_lt_0_10": 0.0,
                    "preferred_estimator": "staggered_did",
                    "preferred_effect": 0.31,
                    "preferred_p_value": 0.01,
                    "identification_strength": 82.0,
                    "effect_sign_consistent_with_main": 1,
                }
            ]
        ).to_csv(DATA_OUTPUTS / "econometric_policy_source_sensitivity.csv", index=False)

        (DATA_OUTPUTS / "benchmark_scores.json").write_text(
            json.dumps(
                {
                    "temporal_holdout": {"linear": {"r2": 0.88}},
                    "spatial_ood": {"mean_linear_r2": 0.76},
                }
            ),
            encoding="utf-8",
        )
        (DATA_OUTPUTS / "pulse_ai_summary.json").write_text(
            json.dumps(
                {
                    "model_metrics": {"roc_auc": 0.79},
                    "dynamic_structure": {
                        "policy_rl": {
                            "offline_policy_evaluation": {
                                "delta_vs_behavior": {"dr": 1.2},
                                "uplift_ci": {"dr": {"delta_ci_low": 0.5, "delta_ci_high": 1.7}},
                                "continent_dr_share_positive": 1.0,
                            }
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        (DATA_OUTPUTS / "external_validity_summary.json").write_text(
            json.dumps({"avg_twfe_abs_t": 2.1, "avg_predictive_r2_uplift": 0.018}),
            encoding="utf-8",
        )
        (DATA_OUTPUTS / "inference_reporting_summary.json").write_text(
            json.dumps({"status": "ok"}),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {"bin": "(0,0.5]", "pred": 0.2, "obs": 0.1, "n": 100},
                {"bin": "(0.5,1]", "pred": 0.8, "obs": 0.7, "n": 100},
            ]
        ).to_csv(DATA_OUTPUTS / "experiment_pulse_calibration_bins.csv", index=False)

        out = run_top_tier_reinforcement_suite()
        self.assertEqual(out.get("status"), "ok")
        self.assertTrue((DATA_OUTPUTS / "top_tier_innovation_frontier.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "top_tier_identification_spectrum.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "top_tier_evidence_convergence.csv").exists())

    def test_submission_extensions_run(self) -> None:
        DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

        out = run_submission_extensions()
        self.assertEqual(out.get("status"), "ok")
        self.assertTrue((DATA_PROCESSED / "policy_event_registry_enriched.csv").exists())
        self.assertTrue((DATA_PROCESSED / "policy_event_source_links.csv").exists())
        self.assertTrue((DATA_PROCESSED / "policy_event_registry_quality_report.json").exists())
        self.assertTrue((DATA_OUTPUTS / "robustness_gate_checks.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "robustness_audit_summary.json").exists())
        self.assertTrue((DATA_OUTPUTS / "reproducibility_manifest.json").exists())
        self.assertTrue((DATA_OUTPUTS / "reproducibility_artifact_hashes.csv").exists())

    def test_identification_plus_suite_runs(self) -> None:
        DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {"variant": "source_external_direct", "design_variant": "direct", "source_channel": "external", "rel_year": -3, "coef": 0.02, "stderr": 0.05, "t_value": 0.4},
                {"variant": "source_external_direct", "design_variant": "direct", "source_channel": "external", "rel_year": -2, "coef": 0.01, "stderr": 0.05, "t_value": 0.2},
                {"variant": "source_external_direct", "design_variant": "direct", "source_channel": "external", "rel_year": 0, "coef": 0.12, "stderr": 0.05, "t_value": 2.4},
                {"variant": "source_external_direct", "design_variant": "direct", "source_channel": "external", "rel_year": 1, "coef": 0.08, "stderr": 0.04, "t_value": 2.0},
                {"variant": "main_design", "design_variant": "main", "source_channel": "main", "rel_year": -2, "coef": -0.01, "stderr": 0.06, "t_value": -0.15},
                {"variant": "main_design", "design_variant": "main", "source_channel": "main", "rel_year": -1, "coef": 0.00, "stderr": 0.06, "t_value": 0.01},
                {"variant": "main_design", "design_variant": "main", "source_channel": "main", "rel_year": 0, "coef": 0.03, "stderr": 0.06, "t_value": 0.5},
                {"variant": "main_design", "design_variant": "main", "source_channel": "main", "rel_year": 1, "coef": 0.04, "stderr": 0.05, "t_value": 0.8},
            ]
        ).to_csv(DATA_OUTPUTS / "econometric_source_event_study_points.csv", index=False)

        pd.DataFrame(
            [
                {
                    "variant": "source_external_direct",
                    "status": "ok",
                    "design_variant": "direct",
                    "did_coef": 0.11,
                    "did_stderr": 0.04,
                    "did_t_value": 2.75,
                    "did_p_value": 0.01,
                    "bootstrap_p_value": 0.03,
                    "permutation_p_value": 0.07,
                    "wild_bootstrap_p_value": 0.06,
                    "lead_placebo_share_p_lt_0_10": 0.0,
                    "identification_strength": 78.0,
                    "effect_sign_consistent_with_main": 1,
                },
                {
                    "variant": "main_design",
                    "status": "ok",
                    "design_variant": "main",
                    "did_coef": 0.03,
                    "did_stderr": 0.05,
                    "did_t_value": 0.6,
                    "did_p_value": 0.55,
                    "bootstrap_p_value": 0.52,
                    "permutation_p_value": 0.58,
                    "wild_bootstrap_p_value": 0.61,
                    "lead_placebo_share_p_lt_0_10": 0.0,
                    "identification_strength": 35.0,
                    "effect_sign_consistent_with_main": 1,
                },
            ]
        ).to_csv(DATA_OUTPUTS / "econometric_policy_source_sensitivity.csv", index=False)

        pd.DataFrame(
            [
                {"excluded_continent": "Asia", "did_coef": 0.08, "causal_st_att": 0.2},
                {"excluded_continent": "Europe", "did_coef": 0.04, "causal_st_att": 0.15},
            ]
        ).to_csv(DATA_OUTPUTS / "experiment_leave_one_continent_out.csv", index=False)

        (DATA_OUTPUTS / "econometric_summary.json").write_text(
            json.dumps(
                {
                    "main_design_variant": "main",
                    "event_study_fe": {"points": [{"rel_year": -2, "coef": -0.02, "stderr": 0.06, "t_value": -0.3}, {"rel_year": -1, "coef": -0.01, "stderr": 0.06, "t_value": -0.2}, {"rel_year": 0, "coef": 0.02, "stderr": 0.05, "t_value": 0.4}]},
                }
            ),
            encoding="utf-8",
        )

        out = run_identification_plus_suite()
        self.assertEqual(out.get("status"), "ok")
        self.assertTrue((DATA_OUTPUTS / "idplus_event_pretrend_geometry.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "idplus_design_concordance_pairs.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "idplus_design_concordance_matrix.csv").exists())
        self.assertTrue((DATA_OUTPUTS / "idplus_identification_stress_index.csv").exists())

    def test_exogenous_shock_suite_runs(self) -> None:
        DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "policy_events_registry.csv"
            pd.DataFrame(
                [
                    {"iso3": "AAA", "start_year": 2018, "end_year": 2018, "policy_intensity": 0.7, "policy_name": "digital_connectivity", "source_ref": "src:a"},
                    {"iso3": "AAA", "start_year": 2020, "end_year": 2020, "policy_intensity": 0.9, "policy_name": "urban_mobility", "source_ref": "src:b"},
                    {"iso3": "BBB", "start_year": 2020, "end_year": 2020, "policy_intensity": 0.8, "policy_name": "energy_transition", "source_ref": "src:c"},
                    {"iso3": "BBB", "start_year": 2022, "end_year": 2022, "policy_intensity": 0.85, "policy_name": "innovation_hub", "source_ref": "src:d"},
                ]
            ).to_csv(registry_path, index=False)

            rows = []
            for cid, iso3, cont, base_dose in [
                ("c1", "AAA", "Asia", 0.8),
                ("c2", "AAA", "Asia", 0.7),
                ("c3", "BBB", "Europe", 0.2),
                ("c4", "BBB", "Europe", 0.1),
            ]:
                for year in range(2017, 2024):
                    treated = 1 if (iso3 == "AAA" and year >= 2020) else 0
                    rows.append(
                        {
                            "city_id": cid,
                            "city_name": cid,
                            "country": iso3,
                            "iso3": iso3,
                            "continent": cont,
                            "year": year,
                            "composite_index": 50 + 0.4 * (year - 2017) + (1.2 if treated else 0.0) + base_dose,
                            "policy_dose_external_direct": base_dose if year <= 2019 else min(1.0, base_dose + 0.1),
                            "policy_intensity_external_direct": base_dose if year <= 2019 else min(1.0, base_dose + 0.1),
                            "did_treatment_external_direct": treated,
                        }
                    )
            panel = pd.DataFrame(rows)

            out = run_exogenous_shock_suite(panel, registry_path=registry_path)
            self.assertEqual(out.get("status"), "ok")
            self.assertTrue((DATA_OUTPUTS / "exoshock_year_index.csv").exists())
            self.assertTrue((DATA_OUTPUTS / "exoshock_event_response.csv").exists())
            self.assertTrue((DATA_OUTPUTS / "exoshock_placebo_year_distribution.csv").exists())

    def test_exogenous_shock_heterogeneity_suite_runs(self) -> None:
        DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            registry_path = Path(tmp) / "policy_events_registry.csv"
            pd.DataFrame(
                [
                    {"iso3": "AAA", "start_year": 2018, "end_year": 2018, "policy_intensity": 0.7, "policy_name": "digital_connectivity", "source_ref": "src:a"},
                    {"iso3": "AAA", "start_year": 2020, "end_year": 2020, "policy_intensity": 0.9, "policy_name": "urban_mobility", "source_ref": "src:b"},
                    {"iso3": "BBB", "start_year": 2020, "end_year": 2020, "policy_intensity": 0.8, "policy_name": "energy_transition", "source_ref": "src:c"},
                    {"iso3": "CCC", "start_year": 2022, "end_year": 2022, "policy_intensity": 0.75, "policy_name": "innovation_hub", "source_ref": "src:d"},
                ]
            ).to_csv(registry_path, index=False)

            # Seed global shock years to exercise the continent/channel diagnostics branch.
            pd.DataFrame(
                [
                    {"year": 2018, "shock_index": 0.2, "is_shock_year": 0},
                    {"year": 2019, "shock_index": 0.3, "is_shock_year": 0},
                    {"year": 2020, "shock_index": 0.9, "is_shock_year": 1},
                    {"year": 2021, "shock_index": 0.4, "is_shock_year": 0},
                    {"year": 2022, "shock_index": 0.85, "is_shock_year": 1},
                    {"year": 2023, "shock_index": 0.35, "is_shock_year": 0},
                ]
            ).to_csv(DATA_OUTPUTS / "exoshock_year_index.csv", index=False)

            rows = []
            for cid, iso3, cont, dose in [
                ("c1", "AAA", "Asia", 0.85),
                ("c2", "AAA", "Asia", 0.75),
                ("c3", "BBB", "Europe", 0.25),
                ("c4", "CCC", "Africa", 0.15),
                ("c5", "CCC", "Africa", 0.20),
            ]:
                for year in range(2017, 2024):
                    shock_bump = 0.8 if year in {2020, 2022} else 0.0
                    rows.append(
                        {
                            "city_id": cid,
                            "city_name": cid,
                            "country": iso3,
                            "iso3": iso3,
                            "continent": cont,
                            "year": year,
                            "composite_index": 49.5 + 0.35 * (year - 2017) + shock_bump + dose,
                            "economic_vitality": 50.0 + 0.40 * (year - 2017) + 0.5 * shock_bump + 0.7 * dose,
                            "livability": 48.5 + 0.25 * (year - 2017) + 0.3 * shock_bump + 0.5 * dose,
                            "innovation": 47.0 + 0.45 * (year - 2017) + 0.7 * shock_bump + 0.9 * dose,
                            "policy_dose_external_direct": dose if year <= 2019 else min(1.0, dose + 0.08),
                            "policy_intensity_external_direct": dose if year <= 2019 else min(1.0, dose + 0.10),
                        }
                    )
            panel = pd.DataFrame(rows)

            out = run_exogenous_shock_heterogeneity_suite(panel, registry_path=registry_path)
            self.assertEqual(out.get("status"), "ok")
            self.assertTrue((DATA_OUTPUTS / "exoshock_policy_type_year_index.csv").exists())
            self.assertTrue((DATA_OUTPUTS / "exoshock_policy_type_event_response.csv").exists())
            self.assertTrue((DATA_OUTPUTS / "exoshock_continent_event_response.csv").exists())
            self.assertTrue((DATA_OUTPUTS / "exoshock_channel_summary.csv").exists())

    def test_data_crawler_skip_all(self) -> None:
        summary = crawl_global_real_sources(
            max_cities=40,
            start_year=2015,
            end_year=2025,
            strict_real_data=True,
            use_cache=True,
            crawl_macro=False,
            crawl_extra_world_bank=False,
            crawl_policy_events=False,
            crawl_weather=False,
            crawl_poi=False,
        )
        self.assertIn("sample", summary)
        self.assertEqual(summary["sample"]["cities"], 40)
        self.assertEqual(summary["sources"]["world_bank_macro"]["status"], "skipped")
        self.assertEqual(summary["sources"]["weather"]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
