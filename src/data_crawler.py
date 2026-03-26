from __future__ import annotations

"""Unified real-data crawler for global city analytics."""

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd

from .city_catalog import load_city_catalog
from .config import TimeRange, load_config
from .gee_city_observed import (
    import_gee_ghsl_city_yearly,
    import_gee_viirs_city_monthly,
    prepare_gee_city_bundle,
)
from .global_data import (
    collect_city_poi,
    collect_city_weather,
    collect_world_bank_panel,
    fetch_world_bank_indicator,
)
from .osm_history_signals import build_city_osm_history_yearly, build_city_poi_yearly_from_ohsome
from .observed_city_signals import build_observed_city_signals
from .policy_event_crawler import build_policy_event_registry
from .social_sentiment import build_city_social_sentiment_yearly
from .utils import DATA_PROCESSED, DATA_RAW, dump_json


EXTRA_WB_INDICATORS: Dict[str, str] = {
    "patent_residents": "IP.PAT.RESD",
    "researchers_per_million": "SP.POP.SCIE.RD.P6",
    "high_tech_exports_share": "TX.VAL.TECH.MF.ZS",
    "employment_rate": "SL.EMP.TOTL.SP.ZS",
    "urban_population_share": "SP.URB.TOTL.IN.ZS",
    "electricity_access": "EG.ELC.ACCS.ZS",
    "fixed_broadband_subscriptions": "IT.NET.BBND.P2",
    "pm25_exposure": "EN.ATM.PM25.MC.M3",
}

EXTRA_WB_CACHE_ALIASES: Dict[str, str] = {
    "patent_residents": "wb_external_patent_residents.csv",
    "researchers_per_million": "wb_external_researchers_per_million.csv",
    "high_tech_exports_share": "wb_external_high_tech_exports_share.csv",
    "employment_rate": "wb_external_employment_rate.csv",
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _skipped(reason: str) -> Dict[str, Any]:
    return {"status": "skipped", "reason": reason}


def _failed(exc: Exception) -> Dict[str, Any]:
    return {"status": "failed", "error": str(exc)}


def _status_from_df(
    df: pd.DataFrame,
    *,
    city_col: str | None = None,
    iso_col: str | None = None,
    year_col: str | None = "year",
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"status": "ok", "rows": int(len(df))}
    if city_col and city_col in df.columns:
        out["cities"] = int(df[city_col].nunique())
    if iso_col and iso_col in df.columns:
        out["iso3_covered"] = int(df[iso_col].nunique())
    if year_col and year_col in df.columns:
        years = pd.to_numeric(df[year_col], errors="coerce").dropna()
        out["years"] = int(years.nunique())
        if not years.empty:
            out["year_range"] = [int(years.min()), int(years.max())]
    return out


def _collect_extra_world_bank_panel(
    cities: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    use_cache: bool,
    cache_only: bool,
) -> Dict[str, Any]:
    iso3_set = set(cities["iso3"].astype(str).str.upper().unique().tolist())
    indicator_meta: Dict[str, Any] = {}
    merged: pd.DataFrame | None = None

    for name, code in EXTRA_WB_INDICATORS.items():
        indicator_path = DATA_RAW / f"wb_extra_{name}.csv"
        frame: pd.DataFrame | None = None
        source = "api"

        candidate_cache_paths = [indicator_path]
        alias_file = EXTRA_WB_CACHE_ALIASES.get(name)
        if alias_file:
            candidate_cache_paths.append(DATA_RAW / alias_file)

        if use_cache:
            for cache_path in candidate_cache_paths:
                if not cache_path.exists():
                    continue
                try:
                    cached = pd.read_csv(cache_path)
                    if {"iso3", "year"}.issubset(cached.columns) and (("value" in cached.columns) or (name in cached.columns)):
                        frame = cached.copy()
                        source = "cache"
                        break
                except Exception:  # noqa: BLE001
                    frame = None

        if frame is None:
            if cache_only:
                indicator_meta[name] = {
                    "status": "skipped",
                    "reason": "cache_only_and_cache_missing",
                    "indicator_code": code,
                    "path": str(indicator_path),
                }
                continue
            try:
                frame = fetch_world_bank_indicator(code, start_year=start_year, end_year=end_year)
                frame.to_csv(indicator_path, index=False)
                source = "api"
            except Exception as exc:  # noqa: BLE001
                indicator_meta[name] = _failed(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
                indicator_meta[name]["indicator_code"] = code
                continue

        frame = frame.copy()
        frame["iso3"] = frame["iso3"].astype(str).str.upper()
        frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
        frame = frame.dropna(subset=["year"]).copy()
        frame["year"] = frame["year"].astype(int)
        frame = frame[
            frame["iso3"].isin(iso3_set) & frame["year"].between(int(start_year), int(end_year))
        ].copy()
        if frame.empty:
            indicator_meta[name] = {
                "status": "empty_overlap",
                "indicator_code": code,
                "source": source,
                "path": str(indicator_path),
            }
            continue

        if name not in frame.columns and "value" in frame.columns:
            frame = frame.rename(columns={"value": name})
        frame = frame[["iso3", "year", name]].copy()

        this_meta = _status_from_df(frame, iso_col="iso3")
        this_meta["indicator_code"] = code
        this_meta["source"] = source
        this_meta["path"] = str(indicator_path)
        indicator_meta[name] = this_meta

        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=["iso3", "year"], how="outer")

    if merged is None:
        return {
            "status": "failed",
            "reason": "no_extra_world_bank_indicator_available",
            "indicator_status": indicator_meta,
        }

    years = list(range(int(start_year), int(end_year) + 1))
    full_index = (
        pd.MultiIndex.from_product([sorted(iso3_set), years], names=["iso3", "year"])
        .to_frame(index=False)
        .copy()
    )
    panel = full_index.merge(merged, on=["iso3", "year"], how="left")
    feature_cols = [c for c in panel.columns if c not in {"iso3", "year"}]
    for col in feature_cols:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
        panel[col] = panel.groupby("iso3")[col].transform(lambda x: x.interpolate(limit_direction="both"))
        median = float(panel[col].median()) if not pd.isna(panel[col].median()) else 0.0
        panel[col] = panel[col].fillna(median)
    panel["external_source"] = "world_bank"

    panel_path = DATA_RAW / "wb_extra_panel.csv"
    panel.to_csv(panel_path, index=False)

    out = _status_from_df(panel, iso_col="iso3")
    out["status"] = "ok"
    out["feature_count"] = int(len(feature_cols))
    out["path"] = str(panel_path)
    out["indicator_status"] = indicator_meta
    return out


def crawl_global_real_sources(
    *,
    max_cities: int = 295,
    start_year: int = 2015,
    end_year: int = 2025,
    strict_real_data: bool = True,
    use_cache: bool = True,
    policy_min_commitment_usd: float = 0.0,
    crawl_macro: bool = True,
    crawl_extra_world_bank: bool = True,
    extra_world_bank_cache_only: bool = False,
    crawl_policy_events: bool = True,
    crawl_weather: bool = True,
    crawl_poi: bool = True,
    crawl_road: bool = False,
    crawl_viirs: bool = False,
    crawl_osm_history: bool = False,
    crawl_poi_yearly: bool = False,
    crawl_social_sentiment: bool = False,
    social_max_records: int = 80,
    road_radius_m: int = 2000,
    historical_viirs_root: str | None = None,
    gee_prepare_bundle: bool = False,
    gee_output_dir: str | None = None,
    gee_asset_id: str = "users/your_username/gee_city_points",
    gee_buffer_m: int = 5000,
    gee_viirs_csv: str | None = None,
    gee_ghsl_csv: str | None = None,
) -> Dict[str, Any]:
    """Crawl real multi-source data and persist raw artifacts."""
    if int(start_year) > int(end_year):
        msg = "start_year must be <= end_year"
        raise ValueError(msg)

    config = load_config()
    config = replace(config, time_range=TimeRange(start_year=int(start_year), end_year=int(end_year)))

    cities = load_city_catalog(max_cities=int(max_cities))
    cities.to_csv(DATA_RAW / "city_catalog.csv", index=False)

    summary: Dict[str, Any] = {
        "status": "ok",
        "generated_at_utc": _utc_now_iso(),
        "run_config": {
            "max_cities": int(max_cities),
            "start_year": int(start_year),
            "end_year": int(end_year),
            "strict_real_data": bool(strict_real_data),
            "use_cache": bool(use_cache),
            "policy_min_commitment_usd": float(policy_min_commitment_usd),
            "crawl_macro": bool(crawl_macro),
            "crawl_extra_world_bank": bool(crawl_extra_world_bank),
            "crawl_policy_events": bool(crawl_policy_events),
            "crawl_weather": bool(crawl_weather),
            "crawl_poi": bool(crawl_poi),
            "crawl_road": bool(crawl_road),
            "crawl_viirs": bool(crawl_viirs),
            "crawl_osm_history": bool(crawl_osm_history),
            "crawl_poi_yearly": bool(crawl_poi_yearly),
            "crawl_social_sentiment": bool(crawl_social_sentiment),
            "social_max_records": int(social_max_records),
            "road_radius_m": int(road_radius_m),
            "historical_viirs_root": str(historical_viirs_root) if historical_viirs_root else None,
            "extra_world_bank_cache_only": bool(extra_world_bank_cache_only),
            "gee_prepare_bundle": bool(gee_prepare_bundle),
            "gee_output_dir": str(gee_output_dir) if gee_output_dir else None,
            "gee_asset_id": str(gee_asset_id),
            "gee_buffer_m": int(gee_buffer_m),
            "gee_viirs_csv": str(gee_viirs_csv) if gee_viirs_csv else None,
            "gee_ghsl_csv": str(gee_ghsl_csv) if gee_ghsl_csv else None,
        },
        "sample": {
            "cities": int(cities["city_id"].nunique()),
            "countries": int(cities["iso3"].nunique()),
            "continents": int(cities["continent"].nunique()),
        },
        "sources": {},
    }

    if gee_prepare_bundle:
        try:
            summary["sources"]["gee_bundle"] = prepare_gee_city_bundle(
                max_cities=int(max_cities),
                buffer_m=int(gee_buffer_m),
                output_dir=gee_output_dir,
                asset_id=str(gee_asset_id),
                start_year=int(start_year),
                end_year=int(end_year),
            )
        except Exception as exc:  # noqa: BLE001
            summary["sources"]["gee_bundle"] = _failed(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            summary["status"] = "partial"
    else:
        summary["sources"]["gee_bundle"] = _skipped("gee_prepare_bundle_disabled")

    if gee_ghsl_csv:
        try:
            summary["sources"]["gee_ghsl"] = import_gee_ghsl_city_yearly(
                source_path=gee_ghsl_csv,
                merge_existing=True,
                max_cities=int(max_cities),
            )
        except Exception as exc:  # noqa: BLE001
            summary["sources"]["gee_ghsl"] = _failed(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            summary["status"] = "partial"
    else:
        summary["sources"]["gee_ghsl"] = _skipped("gee_ghsl_csv_missing")

    if crawl_macro:
        try:
            macro = collect_world_bank_panel(
                cities,
                config,
                use_cache=bool(use_cache),
                strict_real_data=bool(strict_real_data),
            )
            macro_status = _status_from_df(macro, iso_col="iso3")
            macro_status["macro_source_counts"] = macro["macro_source"].value_counts(dropna=False).to_dict()
            macro_status["path"] = str(DATA_RAW / "wb_macro_panel.csv")
            summary["sources"]["world_bank_macro"] = macro_status
        except Exception as exc:  # noqa: BLE001
            summary["sources"]["world_bank_macro"] = _failed(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            summary["status"] = "partial"
    else:
        summary["sources"]["world_bank_macro"] = _skipped("crawl_macro_disabled")

    if crawl_extra_world_bank:
        try:
            extra_status = _collect_extra_world_bank_panel(
                cities,
                start_year=int(start_year),
                end_year=int(end_year),
                use_cache=bool(use_cache),
                cache_only=bool(extra_world_bank_cache_only),
            )
            summary["sources"]["world_bank_extra"] = extra_status
            if extra_status.get("status") != "ok":
                summary["status"] = "partial"
        except Exception as exc:  # noqa: BLE001
            summary["sources"]["world_bank_extra"] = _failed(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            summary["status"] = "partial"
    else:
        summary["sources"]["world_bank_extra"] = _skipped("crawl_extra_world_bank_disabled")

    if crawl_policy_events:
        try:
            policy_meta = build_policy_event_registry(
                cities["iso3"].astype(str).str.upper().unique().tolist(),
                start_year=int(start_year),
                end_year=int(end_year),
                min_commitment_usd=float(policy_min_commitment_usd),
            )
            reg_path = DATA_RAW / "policy_events_registry.csv"
            reg_status: Dict[str, Any] = {
                "status": "ok",
                "path": str(reg_path),
                "crawler_summary": policy_meta,
            }
            if isinstance(policy_meta, dict) and policy_meta.get("status") != "ok":
                if reg_path.exists():
                    reg_status["status"] = "ok_cached_fallback"
                else:
                    reg_status["status"] = "failed"
                    summary["status"] = "partial"
            if reg_path.exists():
                reg = pd.read_csv(reg_path)
                reg_status["event_rows"] = int(len(reg))
                reg_status["iso3_covered"] = int(reg["iso3"].astype(str).str.upper().nunique()) if "iso3" in reg.columns else 0
                reg_status["policy_types"] = int(reg["policy_name"].nunique()) if "policy_name" in reg.columns else 0
            summary["sources"]["policy_events"] = reg_status
        except Exception as exc:  # noqa: BLE001
            summary["sources"]["policy_events"] = _failed(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            summary["status"] = "partial"
    else:
        summary["sources"]["policy_events"] = _skipped("crawl_policy_events_disabled")

    if crawl_weather:
        try:
            weather = collect_city_weather(
                cities,
                config,
                use_cache=bool(use_cache),
                strict_real_data=bool(strict_real_data),
            )
            weather_status = _status_from_df(weather, city_col="city_id")
            weather_status["weather_source_counts"] = weather["weather_source"].value_counts(dropna=False).to_dict()
            weather_status["path"] = str(DATA_RAW / "city_weather_yearly.csv")
            summary["sources"]["weather"] = weather_status
        except Exception as exc:  # noqa: BLE001
            summary["sources"]["weather"] = _failed(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            summary["status"] = "partial"
    else:
        summary["sources"]["weather"] = _skipped("crawl_weather_disabled")

    if crawl_poi:
        try:
            poi = collect_city_poi(
                cities,
                config,
                use_cache=bool(use_cache),
                strict_real_data=bool(strict_real_data),
            )
            poi_status = _status_from_df(poi, city_col="city_id")
            poi_status["poi_source_counts"] = poi["poi_source"].value_counts(dropna=False).to_dict()
            poi_status["path"] = str(DATA_RAW / "city_poi_features.csv")
            summary["sources"]["poi"] = poi_status
        except Exception as exc:  # noqa: BLE001
            summary["sources"]["poi"] = _failed(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            summary["status"] = "partial"
    else:
        summary["sources"]["poi"] = _skipped("crawl_poi_disabled")

    # Optional independent city-level observation layer.
    if crawl_road or crawl_viirs:
        try:
            obs = build_observed_city_signals(
                max_cities=int(max_cities),
                start_year=int(start_year),
                end_year=int(end_year),
                use_cache=bool(use_cache),
                road_radius_m=int(road_radius_m),
                build_road=bool(crawl_road),
                build_viirs=bool(crawl_viirs),
                historical_viirs_root=historical_viirs_root,
                gee_viirs_csv=gee_viirs_csv,
            )
            if crawl_road:
                summary["sources"]["road_network"] = {
                    "status": "ok",
                    "rows": int(obs.road_rows),
                    "path": obs.road_path,
                    "source_counts": obs.road_source_counts,
                }
            else:
                summary["sources"]["road_network"] = _skipped("crawl_road_disabled")

            if crawl_viirs:
                summary["sources"]["viirs_nightlight"] = {
                    "status": "ok",
                    "monthly_rows": int(obs.viirs_monthly_rows),
                    "daily_rows": int(obs.viirs_daily_rows),
                    "historical_rows": int(obs.viirs_historical_rows),
                    "monthly_path": obs.viirs_monthly_path,
                    "daily_path": obs.viirs_daily_path,
                    "source_counts": obs.viirs_source_counts,
                }
            else:
                summary["sources"]["viirs_nightlight"] = _skipped("crawl_viirs_disabled")
        except Exception as exc:  # noqa: BLE001
            if crawl_road:
                summary["sources"]["road_network"] = _failed(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            else:
                summary["sources"]["road_network"] = _skipped("crawl_road_disabled")
            if crawl_viirs:
                summary["sources"]["viirs_nightlight"] = _failed(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            else:
                summary["sources"]["viirs_nightlight"] = _skipped("crawl_viirs_disabled")
            summary["status"] = "partial"
    else:
        summary["sources"]["road_network"] = _skipped("crawl_road_disabled")
        summary["sources"]["viirs_nightlight"] = _skipped("crawl_viirs_disabled")

    if gee_viirs_csv and not (crawl_road or crawl_viirs):
        try:
            summary["sources"]["gee_viirs_import"] = import_gee_viirs_city_monthly(
                source_path=gee_viirs_csv,
                merge_existing=True,
                max_cities=int(max_cities),
            )
        except Exception as exc:  # noqa: BLE001
            summary["sources"]["gee_viirs_import"] = _failed(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            summary["status"] = "partial"
    elif gee_viirs_csv:
        summary["sources"]["gee_viirs_import"] = {"status": "ok", "mode": "merged_via_observed_city_signals"}
    else:
        summary["sources"]["gee_viirs_import"] = _skipped("gee_viirs_csv_missing")

    if crawl_osm_history:
        try:
            osm_hist = build_city_osm_history_yearly(
                max_cities=int(max_cities),
                start_year=int(start_year),
                end_year=int(end_year),
                use_cache=bool(use_cache),
            )
            summary["sources"]["osm_history"] = {
                "status": "ok",
                "rows": int(len(osm_hist)),
                "cities": int(osm_hist["city_id"].nunique()) if "city_id" in osm_hist.columns else 0,
                "path": str(DATA_RAW / "city_osm_history_yearly.csv"),
                "source_counts": osm_hist.get("osm_hist_source", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
                if len(osm_hist)
                else {},
            }
        except Exception as exc:  # noqa: BLE001
            summary["sources"]["osm_history"] = _failed(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            summary["status"] = "partial"
    else:
        summary["sources"]["osm_history"] = _skipped("crawl_osm_history_disabled")

    if crawl_poi_yearly:
        try:
            poi_yearly = build_city_poi_yearly_from_ohsome(
                max_cities=int(max_cities),
                start_year=int(start_year),
                end_year=int(end_year),
                use_cache=bool(use_cache),
            )
            summary["sources"]["poi_yearly"] = {
                "status": "ok",
                "rows": int(len(poi_yearly)),
                "cities": int(poi_yearly["city_id"].nunique()) if "city_id" in poi_yearly.columns else 0,
                "path": str(DATA_RAW / "city_poi_features_yearly.csv"),
                "temporal_source_counts": poi_yearly.get("poi_temporal_source", pd.Series(dtype=str)).value_counts(dropna=False).to_dict()
                if len(poi_yearly)
                else {},
            }
        except Exception as exc:  # noqa: BLE001
            summary["sources"]["poi_yearly"] = _failed(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            summary["status"] = "partial"
    else:
        summary["sources"]["poi_yearly"] = _skipped("crawl_poi_yearly_disabled")

    if crawl_social_sentiment:
        try:
            social_status = build_city_social_sentiment_yearly(
                max_cities=int(max_cities),
                start_year=int(start_year),
                end_year=int(end_year),
                use_cache=bool(use_cache),
                max_records=int(social_max_records),
            )
            summary["sources"]["social_sentiment"] = social_status
            if social_status.get("status") not in {"ok", "ok_cached"}:
                summary["status"] = "partial"
        except Exception as exc:  # noqa: BLE001
            summary["sources"]["social_sentiment"] = _failed(exc if isinstance(exc, Exception) else RuntimeError(str(exc)))
            summary["status"] = "partial"
    else:
        summary["sources"]["social_sentiment"] = _skipped("crawl_social_sentiment_disabled")

    dump_json(DATA_PROCESSED / "data_crawl_summary.json", summary)
    return summary
