from __future__ import annotations

"""City-level representation learning utilities."""

import logging
from typing import List

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .utils import DATA_OUTPUTS

LOGGER = logging.getLogger(__name__)


REP_FEATURES: List[str] = [
    "composite_index",
    "economic_vitality",
    "livability",
    "innovation",
    "log_gdp_pc",
    "gdp_growth",
    "unemployment",
    "internet_users",
    "capital_formation",
    "inflation",
    "digital_infra_n",
    "knowledge_capital_n",
    "clean_air_n",
    "basic_infra_n",
    "labor_market_n",
    "temperature_mean",
    "precipitation_sum",
    "poi_total",
    "poi_diversity",
]


def _city_trend(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    x = pd.Series(range(len(series)), dtype=float)
    y = pd.to_numeric(series, errors="coerce").fillna(series.mean())
    x_center = x - x.mean()
    denom = float((x_center**2).sum())
    if denom <= 1e-12:
        return 0.0
    num = float(((y - y.mean()) * x_center).sum())
    return num / denom


def build_city_embeddings(panel: pd.DataFrame, n_components: int = 8) -> pd.DataFrame:
    """Build compact city embeddings from multi-source panel features."""
    use_cols = [c for c in REP_FEATURES if c in panel.columns]
    if not use_cols:
        raise RuntimeError("No representation features available.")

    grouped = panel.groupby("city_id", as_index=False)
    city_base = grouped[["city_name", "country", "continent", "latitude", "longitude"]].first()

    mean_features = grouped[use_cols].mean(numeric_only=True)
    std_features = grouped[["composite_index"]].std(numeric_only=True).rename(
        columns={"composite_index": "composite_volatility"}
    )
    trend_features = (
        panel.sort_values(["city_id", "year"])
        .groupby("city_id")["composite_index"]
        .apply(_city_trend)
        .reset_index(name="composite_trend")
    )

    feats = city_base.merge(mean_features, on="city_id", how="left")
    feats = feats.merge(std_features, on="city_id", how="left")
    feats = feats.merge(trend_features, on="city_id", how="left")

    numeric_cols = [c for c in feats.columns if c not in {"city_id", "city_name", "country", "continent"}]
    x = feats[numeric_cols].copy()
    x = x.fillna(x.median(numeric_only=True))

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    k = max(2, min(n_components, x_scaled.shape[0] - 1, x_scaled.shape[1]))
    pca = PCA(n_components=k, random_state=42)
    z = pca.fit_transform(x_scaled)

    emb = feats[["city_id", "city_name", "country", "continent"]].copy()
    for i in range(z.shape[1]):
        emb[f"emb_{i+1}"] = z[:, i]

    emb.to_csv(DATA_OUTPUTS / "city_embeddings.csv", index=False)

    var_ratio = float(pca.explained_variance_ratio_.sum()) if hasattr(pca, "explained_variance_ratio_") else 0.0
    LOGGER.info("City embeddings built: %s cities, %s dims, explained_var=%.3f", len(emb), z.shape[1], var_ratio)
    return emb
