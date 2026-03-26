from __future__ import annotations

"""Build cross-sectional city aviation connectivity features from OpenFlights."""

import csv
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import DATA_PROCESSED, DATA_RAW, dump_json, haversine_km


AIRPORT_COLUMNS = [
    "airport_id",
    "airport_name",
    "airport_city",
    "airport_country",
    "iata",
    "icao",
    "latitude",
    "longitude",
    "altitude_ft",
    "utc_offset_hours",
    "dst",
    "timezone",
    "type",
    "source",
]

ROUTE_COLUMNS = [
    "airline",
    "airline_id",
    "source_airport",
    "source_airport_id",
    "dest_airport",
    "dest_airport_id",
    "codeshare",
    "stops",
    "equipment",
]

COUNTRY_ALIASES = {
    "republic of the congo": "congo brazzaville",
    "democratic republic of the congo": "congo kinshasa",
    "czechia": "czech republic",
    "viet nam": "vietnam",
}

CITY_ALIAS_MAP = {
    "algiers_dz": {"algier", "alger"},
    "almaty_kz": {"alma ata"},
    "ashgabat_tm": {"ashkhabad"},
    "astana_kz": {"nur sultan", "nursultan", "tselinograd", "aqmola"},
    "chennai_in": {"madras"},
    "cusco_pe": {"cuzco"},
    "dusseldorf_de": {"duesseldorf"},
    "gaborone_bw": {"gaberone"},
    "gold_coast_au": {"coolangatta"},
    "incheon_kr": {"incheon"},
    "kyiv_ua": {"kiev"},
    "lae_pg": {"nadzab"},
    "leon_mx": {"del bajio", "del bajio international", "bajio"},
    "nouakchott_mr": {"nouakschott"},
    "port_harcourt_ng": {"port hartcourt"},
    "seville_es": {"sevilla"},
    "suva_fj": {"nausori"},
    "ulaanbaatar_mn": {"ulan bator", "ulaan baatar"},
    "valparaiso_cl": {"vina del mar", "vina"},
}

NAME_MATCH_MAX_DISTANCE_KM = 60.0
NEAREST_FALLBACK_MAX_DISTANCE_KM = 45.0


def _normalize_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_country(value: object) -> str:
    base = _normalize_text(value)
    return COUNTRY_ALIASES.get(base, base)


def _city_aliases(city_id: str, city_name: str) -> set[str]:
    aliases = {_normalize_text(city_name)}
    aliases.update(_normalize_text(v) for v in CITY_ALIAS_MAP.get(city_id, set()))
    return {v for v in aliases if v}


def _load_city_catalog() -> pd.DataFrame:
    cities = pd.read_csv(DATA_RAW / "city_catalog.csv")
    cols = ["city_id", "city_name", "country", "iso3", "latitude", "longitude"]
    cities = cities[cols].copy()
    cities["country_key"] = cities["country"].map(_normalize_country)
    cities["city_key"] = cities["city_name"].map(_normalize_text)
    cities["city_aliases"] = [
        sorted(_city_aliases(city_id, city_name))
        for city_id, city_name in zip(cities["city_id"].astype(str), cities["city_name"].astype(str), strict=False)
    ]
    return cities


def _load_airports(path: Path) -> pd.DataFrame:
    airports = pd.read_csv(path, header=None, names=AIRPORT_COLUMNS, dtype=str, keep_default_na=False, na_filter=False)
    airports["latitude"] = pd.to_numeric(airports["latitude"], errors="coerce")
    airports["longitude"] = pd.to_numeric(airports["longitude"], errors="coerce")
    airports = airports.dropna(subset=["airport_id", "latitude", "longitude"]).copy()
    airports["airport_id"] = airports["airport_id"].astype(str)
    airports["airport_city_key"] = airports["airport_city"].map(_normalize_text)
    airports["airport_name_key"] = airports["airport_name"].map(_normalize_text)
    airports["country_key"] = airports["airport_country"].map(_normalize_country)
    return airports


def _assign_airports_to_cities(cities: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    city_by_country: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in cities.itertuples(index=False):
        city_by_country[str(row.country_key)].append(
            {
                "city_id": str(row.city_id),
                "city_name": str(row.city_name),
                "iso3": str(row.iso3),
                "country": str(row.country),
                "country_key": str(row.country_key),
                "latitude": float(row.latitude),
                "longitude": float(row.longitude),
                "aliases": set(row.city_aliases),
            }
        )

    assignments: list[dict[str, object]] = []
    for airport in airports.itertuples(index=False):
        candidates = city_by_country.get(str(airport.country_key), [])
        if not candidates:
            continue
        best: dict[str, object] | None = None
        for city in candidates:
            distance_km = haversine_km(
                float(airport.latitude),
                float(airport.longitude),
                float(city["latitude"]),
                float(city["longitude"]),
            )
            aliases = city["aliases"]
            airport_city_key = str(airport.airport_city_key)
            airport_name_key = str(airport.airport_name_key)
            if airport_city_key in aliases:
                priority = 0
                method = "airport_city_exact"
            elif any(alias and alias in airport_name_key for alias in aliases):
                priority = 1
                method = "airport_name_alias"
            elif distance_km <= NEAREST_FALLBACK_MAX_DISTANCE_KM:
                priority = 2
                method = "nearest_city_fallback"
            else:
                continue

            if method != "nearest_city_fallback" and distance_km > NAME_MATCH_MAX_DISTANCE_KM:
                continue

            cand = {
                "airport_id": str(airport.airport_id),
                "airport_name": str(airport.airport_name),
                "airport_city": str(airport.airport_city),
                "airport_country": str(airport.airport_country),
                "iata": str(airport.iata),
                "icao": str(airport.icao),
                "latitude": float(airport.latitude),
                "longitude": float(airport.longitude),
                "city_id": str(city["city_id"]),
                "city_name": str(city["city_name"]),
                "iso3": str(city["iso3"]),
                "distance_km": float(distance_km),
                "mapping_method": method,
                "_priority": int(priority),
            }
            if best is None or (cand["_priority"], cand["distance_km"], cand["city_id"]) < (
                best["_priority"],
                best["distance_km"],
                best["city_id"],
            ):
                best = cand
        if best is not None:
            assignments.append(best)

    mapped = pd.DataFrame(assignments)
    if mapped.empty:
        return mapped
    mapped = mapped.sort_values(["airport_id", "_priority", "distance_km", "city_id"]).drop_duplicates("airport_id", keep="first")
    return mapped.drop(columns="_priority").reset_index(drop=True)


def _load_routes(path: Path) -> pd.DataFrame:
    routes = pd.read_csv(path, header=None, names=ROUTE_COLUMNS, dtype=str, keep_default_na=False, na_filter=False)
    routes = routes[(routes["source_airport_id"] != "\\N") & (routes["dest_airport_id"] != "\\N")].copy()
    routes["source_airport_id"] = routes["source_airport_id"].astype(str)
    routes["dest_airport_id"] = routes["dest_airport_id"].astype(str)
    return routes


def build_openflights_connectivity(
    airports_path: Path | None = None,
    routes_path: Path | None = None,
) -> dict[str, object]:
    airports_path = airports_path or (DATA_RAW / "openflights" / "airports.dat")
    routes_path = routes_path or (DATA_RAW / "openflights" / "routes.dat")
    if not airports_path.exists():
        raise FileNotFoundError(airports_path)
    if not routes_path.exists():
        raise FileNotFoundError(routes_path)

    cities = _load_city_catalog()
    airports = _load_airports(airports_path)
    routes = _load_routes(routes_path)
    mapped_airports = _assign_airports_to_cities(cities, airports)
    mapped_airports.to_csv(DATA_RAW / "openflights_airport_city_mapping.csv", index=False)
    if mapped_airports.empty:
        raise RuntimeError("no_airports_mapped_to_project_cities")

    airport_country = airports.set_index("airport_id")["country_key"].to_dict()
    mapped_lookup = mapped_airports.set_index("airport_id")[["city_id", "iso3"]].to_dict(orient="index")
    city_country_key = cities.set_index("city_id")["country_key"].to_dict()
    active_airports = set(routes["source_airport_id"].astype(str)).union(set(routes["dest_airport_id"].astype(str)))

    city_route_total: Counter[str] = Counter()
    city_route_outbound: Counter[str] = Counter()
    city_route_inbound: Counter[str] = Counter()
    city_partner_airports: defaultdict[str, set[str]] = defaultdict(set)
    city_partner_cities: defaultdict[str, set[str]] = defaultdict(set)
    city_international_rows: Counter[str] = Counter()
    city_active_airports: defaultdict[str, set[str]] = defaultdict(set)
    edge_counter: Counter[tuple[str, str]] = Counter()
    airport_route_total: Counter[str] = Counter()
    airport_route_outbound: Counter[str] = Counter()
    airport_route_inbound: Counter[str] = Counter()
    airport_partner_airports: defaultdict[str, set[str]] = defaultdict(set)
    airport_international_rows: Counter[str] = Counter()

    for row in routes.itertuples(index=False):
        src_id = str(row.source_airport_id)
        dst_id = str(row.dest_airport_id)
        src_meta = mapped_lookup.get(src_id)
        dst_meta = mapped_lookup.get(dst_id)
        src_country = airport_country.get(src_id)
        dst_country = airport_country.get(dst_id)

        airport_route_total[src_id] += 1
        airport_route_outbound[src_id] += 1
        airport_partner_airports[src_id].add(dst_id)
        if src_country != dst_country:
            airport_international_rows[src_id] += 1

        airport_route_total[dst_id] += 1
        airport_route_inbound[dst_id] += 1
        airport_partner_airports[dst_id].add(src_id)
        if src_country != dst_country:
            airport_international_rows[dst_id] += 1

        if src_meta is not None:
            src_city = str(src_meta["city_id"])
            city_route_total[src_city] += 1
            city_route_outbound[src_city] += 1
            city_partner_airports[src_city].add(dst_id)
            city_active_airports[src_city].add(src_id)
            if airport_country.get(dst_id) != city_country_key.get(src_city):
                city_international_rows[src_city] += 1
            if dst_meta is not None and str(dst_meta["city_id"]) != src_city:
                city_partner_cities[src_city].add(str(dst_meta["city_id"]))
                edge_counter[(src_city, str(dst_meta["city_id"]))] += 1

        if dst_meta is not None:
            dst_city = str(dst_meta["city_id"])
            city_route_total[dst_city] += 1
            city_route_inbound[dst_city] += 1
            city_partner_airports[dst_city].add(src_id)
            city_active_airports[dst_city].add(dst_id)
            if airport_country.get(src_id) != city_country_key.get(dst_city):
                city_international_rows[dst_city] += 1
            if src_meta is not None and str(src_meta["city_id"]) != dst_city:
                city_partner_cities[dst_city].add(str(src_meta["city_id"]))

    airport_counts = mapped_airports.groupby("city_id")["airport_id"].nunique().to_dict()
    active_airport_count = max(len(active_airports), 1)
    city_coords = cities.set_index("city_id")[["latitude", "longitude"]].to_dict(orient="index")
    rows: list[dict[str, object]] = []
    for row in cities.itertuples(index=False):
        city_id = str(row.city_id)
        total = int(city_route_total.get(city_id, 0))
        partner_count = int(len(city_partner_airports.get(city_id, set())))
        rows.append(
            {
                "city_id": city_id,
                "city_name": str(row.city_name),
                "iso3": str(row.iso3),
                "flight_connectivity_total": float(total),
                "flight_degree_centrality": float(partner_count / active_airport_count),
                "airport_count_mapped": int(airport_counts.get(city_id, 0)),
                "airport_count_active": int(len(city_active_airports.get(city_id, set()))),
                "flight_route_rows_outbound": int(city_route_outbound.get(city_id, 0)),
                "flight_route_rows_inbound": int(city_route_inbound.get(city_id, 0)),
                "international_route_share": float(city_international_rows.get(city_id, 0) / total) if total else 0.0,
                "mapped_partner_city_count": int(len(city_partner_cities.get(city_id, set()))),
                "shared_airport_fallback_used": 0,
                "shared_airport_iata": "",
                "shipping_connectivity_total": np.nan,
                "network_connectivity_source": "openflights_static_routes_cross_section",
            }
        )
    connectivity = pd.DataFrame(rows).sort_values(["flight_connectivity_total", "flight_degree_centrality"], ascending=[False, False])

    airport_geo = airports.set_index("airport_id")[["latitude", "longitude", "country_key", "iata"]].to_dict(orient="index")
    active_airport_geo = {
        airport_id: airport_geo[airport_id]
        for airport_id in active_airports
        if airport_id in airport_geo
    }
    for idx, row in connectivity.loc[connectivity["flight_connectivity_total"] <= 0].iterrows():
        city_geo = city_coords.get(str(row["city_id"]), {})
        best_airport_id = None
        best_score = None
        for airport_id, meta in active_airport_geo.items():
            distance_km = haversine_km(
                float(city_geo.get("latitude", 0.0)),
                float(city_geo.get("longitude", 0.0)),
                float(meta["latitude"]),
                float(meta["longitude"]),
            )
            same_country = int(str(meta["country_key"]) == city_country_key.get(str(row["city_id"])))
            if same_country and distance_km <= 120.0:
                score = (0, distance_km, str(airport_id))
            elif distance_km <= 60.0:
                score = (1, distance_km, str(airport_id))
            else:
                continue
            if best_score is None or score < best_score:
                best_score = score
                best_airport_id = str(airport_id)
        if best_airport_id is None:
            continue
        total = int(airport_route_total.get(best_airport_id, 0))
        partners = int(len(airport_partner_airports.get(best_airport_id, set())))
        connectivity.loc[idx, "flight_connectivity_total"] = float(total)
        connectivity.loc[idx, "flight_degree_centrality"] = float(partners / active_airport_count)
        connectivity.loc[idx, "airport_count_active"] = max(int(connectivity.loc[idx, "airport_count_active"]), 1)
        connectivity.loc[idx, "flight_route_rows_outbound"] = int(airport_route_outbound.get(best_airport_id, 0))
        connectivity.loc[idx, "flight_route_rows_inbound"] = int(airport_route_inbound.get(best_airport_id, 0))
        connectivity.loc[idx, "international_route_share"] = (
            float(airport_international_rows.get(best_airport_id, 0) / total) if total else 0.0
        )
        connectivity.loc[idx, "shared_airport_fallback_used"] = 1
        connectivity.loc[idx, "shared_airport_iata"] = str(active_airport_geo.get(best_airport_id, {}).get("iata", ""))
        connectivity.loc[idx, "network_connectivity_source"] = "openflights_static_routes_shared_airport_fallback"

    connectivity.to_csv(DATA_RAW / "city_connectivity.csv", index=False)

    edge_rows = [
        {
            "source_city_id": src,
            "target_city_id": dst,
            "route_rows": int(weight),
        }
        for (src, dst), weight in sorted(edge_counter.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    ]
    pd.DataFrame(edge_rows, columns=["source_city_id", "target_city_id", "route_rows"]).to_csv(
        DATA_PROCESSED / "openflights_city_route_edges.csv",
        index=False,
    )

    unmatched = cities.loc[~cities["city_id"].isin(connectivity.loc[connectivity["airport_count_mapped"] > 0, "city_id"])].copy()
    unmatched[["city_id", "city_name", "iso3", "country"]].to_csv(DATA_RAW / "openflights_city_unmatched.csv", index=False)

    summary = {
        "status": "ok",
        "source_airports_path": str(airports_path),
        "source_routes_path": str(routes_path),
        "cities_total": int(len(cities)),
        "cities_with_mapped_airports": int((connectivity["airport_count_mapped"] > 0).sum()),
        "cities_with_active_routes": int((connectivity["flight_connectivity_total"] > 0).sum()),
        "mapped_airports_total": int(len(mapped_airports)),
        "mapped_active_airports": int(mapped_airports["airport_id"].astype(str).isin(active_airports).sum()),
        "route_rows_total": int(len(routes)),
        "mapping_method_counts": mapped_airports["mapping_method"].value_counts().to_dict(),
        "top_connected_cities": connectivity.head(20)[
            ["city_id", "flight_connectivity_total", "flight_degree_centrality", "mapped_partner_city_count"]
        ].to_dict(orient="records"),
        "unmatched_cities": unmatched["city_id"].astype(str).tolist(),
    }
    dump_json(DATA_PROCESSED / "openflights_connectivity_summary.json", summary)
    return summary


if __name__ == "__main__":
    result = build_openflights_connectivity()
    print(result)
