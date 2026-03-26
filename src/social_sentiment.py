from __future__ import annotations

"""Social-sentiment collection and aggregation utilities.

This module provides two layers:
1) A lightweight lexicon scorer for text sentiment.
2) A city-year crawler/aggregator using GDELT Doc API over social-media domains.

Outputs are stored in data/raw as:
- social_sentiment_posts.csv (post/article-level records)
- city_social_sentiment_yearly.csv (city-year aggregated indicators)
"""

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import requests

from .city_catalog import load_city_catalog
from .utils import DATA_PROCESSED, DATA_RAW, dump_json

LOGGER = logging.getLogger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_SOCIAL_DOMAINS: Sequence[str] = (
    "reddit.com",
    "x.com",
    "twitter.com",
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "tiktok.com",
    "weibo.com",
)

POSITIVE_WORDS = {
    "good",
    "great",
    "excellent",
    "positive",
    "improve",
    "improved",
    "improving",
    "growth",
    "growing",
    "booming",
    "strong",
    "record",
    "opportunity",
    "efficient",
    "innovation",
    "resilient",
    "safe",
    "clean",
    "vibrant",
    "affordable",
    "optimistic",
    "恢复",
    "改善",
    "增长",
    "繁荣",
    "稳定",
    "创新",
    "便利",
    "安全",
    "宜居",
}

NEGATIVE_WORDS = {
    "bad",
    "worse",
    "worst",
    "negative",
    "decline",
    "declining",
    "drop",
    "collapse",
    "crisis",
    "risk",
    "risky",
    "unemployment",
    "pollution",
    "crime",
    "congestion",
    "expensive",
    "shortage",
    "fragile",
    "volatile",
    "stagnation",
    "衰退",
    "风险",
    "危机",
    "失业",
    "污染",
    "拥堵",
    "高成本",
    "下滑",
    "停滞",
}

TOKEN_RE = re.compile(r"[A-Za-z]+|[\u4e00-\u9fff]{1,4}")


def simple_sentiment_score(text: str | None) -> float:
    """Compute a lightweight lexicon sentiment score in [-1, 1]."""
    if text is None:
        return 0.0
    content = str(text).strip().lower()
    if not content:
        return 0.0

    tokens = TOKEN_RE.findall(content)
    if not tokens:
        return 0.0

    pos = sum(1 for t in tokens if t in POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0

    score = (pos - neg) / float(total)
    return float(np.clip(score, -1.0, 1.0))


def _normalize_gdelt_tone(value: Any) -> float | None:
    """Map GDELT tone-like values to [-1, 1] when available."""
    try:
        tone = float(value)
    except Exception:  # noqa: BLE001
        return None
    if not np.isfinite(tone):
        return None
    scale = 100.0 if abs(tone) > 20.0 else 10.0
    return float(np.clip(tone / scale, -1.0, 1.0))


def _attach_city_id_from_name(df: pd.DataFrame, cities: pd.DataFrame) -> pd.DataFrame:
    if "city_id" in df.columns:
        out = df.copy()
        out["city_id"] = out["city_id"].astype(str)
        return out
    if "city_name" not in df.columns:
        return df.copy()

    mapper = cities[["city_id", "city_name"]].copy()
    mapper["city_name_l"] = mapper["city_name"].astype(str).str.strip().str.lower()
    out = df.copy()
    out["city_name_l"] = out["city_name"].astype(str).str.strip().str.lower()
    out = out.merge(mapper[["city_id", "city_name_l"]], on="city_name_l", how="left")
    return out.drop(columns=["city_name_l"], errors="ignore")


def _resolve_text_column(df: pd.DataFrame) -> str | None:
    for col in ["text", "content", "title", "headline", "body", "message", "post_text"]:
        if col in df.columns:
            return col
    return None


def _resolve_date_column(df: pd.DataFrame) -> str | None:
    for col in ["date", "datetime", "created_at", "timestamp", "time", "seendate"]:
        if col in df.columns:
            return col
    return None


def aggregate_social_posts(
    posts: pd.DataFrame,
    cities: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    source_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate social/news posts into city-year sentiment panel.

    Returns:
    - panel: city-year aggregated sentiment indicators.
    - post_std: standardized post-level frame used for aggregation.
    """
    if posts.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = _attach_city_id_from_name(posts, cities)
    if "city_id" not in work.columns:
        return pd.DataFrame(), pd.DataFrame()

    work["city_id"] = work["city_id"].astype(str)
    work = work[work["city_id"].isin(cities["city_id"].astype(str))].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "year" not in work.columns:
        date_col = _resolve_date_column(work)
        if date_col:
            work["year"] = pd.to_datetime(work[date_col], errors="coerce").dt.year
    work["year"] = pd.to_numeric(work.get("year"), errors="coerce")
    work = work[work["year"].between(int(start_year), int(end_year), inclusive="both")].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work["year"] = work["year"].astype(int)

    if "sentiment_score" in work.columns:
        work["sentiment_score"] = pd.to_numeric(work["sentiment_score"], errors="coerce")
    else:
        text_col = _resolve_text_column(work)
        if text_col is None:
            work["sentiment_score"] = 0.0
        else:
            work["sentiment_score"] = work[text_col].map(simple_sentiment_score).astype(float)

    # Keep only finite scores for stable aggregation.
    work = work[np.isfinite(work["sentiment_score"].to_numpy(dtype=float))].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "platform" not in work.columns:
        for fallback in ["domain", "source", "source_domain", "source_platform"]:
            if fallback in work.columns:
                work["platform"] = work[fallback].astype(str)
                break
        else:
            work["platform"] = "unknown"

    def _score_std(s: pd.Series) -> float:
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return float(0.0)
        return float(np.std(arr, ddof=0))

    agg = (
        work.groupby(["city_id", "year"], as_index=False)
        .agg(
            social_sentiment_score=("sentiment_score", "mean"),
            social_sentiment_volatility=("sentiment_score", _score_std),
            social_sentiment_positive_share=("sentiment_score", lambda s: float((pd.to_numeric(s, errors="coerce") > 0.15).mean())),
            social_sentiment_negative_share=("sentiment_score", lambda s: float((pd.to_numeric(s, errors="coerce") < -0.15).mean())),
            social_sentiment_volume=("sentiment_score", "size"),
            social_sentiment_platform_count=("platform", "nunique"),
        )
        .copy()
    )
    agg["social_sentiment_buzz"] = np.log1p(pd.to_numeric(agg["social_sentiment_volume"], errors="coerce").fillna(0.0))
    if "social_sentiment_source" in work.columns:
        src = (
            work.groupby(["city_id", "year"], as_index=False)["social_sentiment_source"]
            .agg(lambda s: str(pd.Series(s).dropna().astype(str).iloc[-1]) if not pd.Series(s).dropna().empty else str(source_label))
        )
        agg = agg.merge(src, on=["city_id", "year"], how="left")
        agg["social_sentiment_source"] = agg["social_sentiment_source"].fillna(str(source_label)).astype(str)
    else:
        agg["social_sentiment_source"] = str(source_label)

    city_lookup = cities[["city_id", "city_name"]].drop_duplicates("city_id").copy()
    post_std = work.merge(city_lookup, on="city_id", how="left", suffixes=("", "_catalog"))
    if "city_name_catalog" in post_std.columns:
        post_std["city_name"] = post_std["city_name_catalog"].fillna(post_std.get("city_name"))
        post_std = post_std.drop(columns=["city_name_catalog"], errors="ignore")

    keep_cols = [
        c
        for c in [
            "city_id",
            "city_name",
            "year",
            "platform",
            "sentiment_score",
            "date",
            "datetime",
            "created_at",
            "timestamp",
            "seendate",
            "text",
            "title",
            "content",
            "headline",
            "body",
            "message",
            "post_text",
            "source",
            "domain",
            "url",
            "social_sentiment_source",
            "tone",
        ]
        if c in post_std.columns
    ]
    post_std = post_std[keep_cols].copy()

    return agg, post_std


def _gdelt_datetime(year: int, is_end: bool = False) -> str:
    if is_end:
        return f"{int(year)}1231235959"
    return f"{int(year)}0101000000"


def _gdelt_social_query(city_name: str, domains: Sequence[str]) -> str:
    domain_clause = " OR ".join([f"domainis:{d}" for d in domains])
    return f'("{city_name}" OR "{city_name} city") AND ({domain_clause})'


def _gdelt_city_news_query(city_name: str, country: str | None = None) -> str:
    country_text = str(country or "").strip()
    if country_text:
        return f'"{city_name}" AND "{country_text}"'
    return f'"{city_name}"'


def _gdelt_queries_for_city(
    city_name: str,
    country: str | None,
    domains: Sequence[str],
) -> list[tuple[str, str]]:
    queries = [
        ("gdelt_social_media_proxy", _gdelt_social_query(city_name, domains)),
        ("gdelt_city_news_proxy", _gdelt_city_news_query(city_name, country)),
    ]
    if str(country or "").strip():
        queries.append(("gdelt_city_news_proxy", f'"{city_name}"'))
    return queries


def _fetch_gdelt_articles(
    query: str,
    *,
    start_dt: str,
    end_dt: str,
    max_records: int,
    timeout: int,
    retries: int = 4,
    backoff_seconds: float = 2.0,
) -> List[dict]:
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": int(max_records),
        "sort": "DateDesc",
        "STARTDATETIME": start_dt,
        "ENDDATETIME": end_dt,
    }
    last_error: Exception | None = None
    payload: dict | None = None
    for attempt in range(1, int(retries) + 1):
        try:
            resp = requests.get(GDELT_DOC_API, params=params, timeout=int(timeout))
            if resp.status_code in {429, 500, 502, 503, 504}:
                retry_after_raw = resp.headers.get("Retry-After")
                retry_after = 0.0
                try:
                    retry_after = float(retry_after_raw) if retry_after_raw is not None else 0.0
                except Exception:  # noqa: BLE001
                    retry_after = 0.0
                if attempt < int(retries):
                    sleep_s = max(float(backoff_seconds) * (2 ** (attempt - 1)), retry_after)
                    time.sleep(max(0.0, sleep_s))
                resp.raise_for_status()
            resp.raise_for_status()
            payload = resp.json()
            last_error = None
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= int(retries):
                break
            sleep_s = float(backoff_seconds) * (2 ** (attempt - 1))
            time.sleep(max(0.0, sleep_s))
    if last_error is not None:
        raise RuntimeError(str(last_error))
    if not isinstance(payload, dict):
        return []
    art = payload.get("articles", [])
    if not isinstance(art, list):
        return []
    return [a for a in art if isinstance(a, dict)]


def _score_gdelt_article(article: dict) -> float:
    tone = _normalize_gdelt_tone(article.get("tone"))
    if tone is not None:
        return tone
    return simple_sentiment_score(article.get("title"))


def _standardize_cached_social_yearly(
    cached: pd.DataFrame,
    cities: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    if cached.empty:
        return pd.DataFrame()
    work = _attach_city_id_from_name(cached, cities)
    if "city_id" not in work.columns or "year" not in work.columns:
        return pd.DataFrame()
    work["city_id"] = work["city_id"].astype(str)
    work["year"] = pd.to_numeric(work["year"], errors="coerce")
    work = work[work["city_id"].isin(cities["city_id"].astype(str))].copy()
    work = work[work["year"].between(int(start_year), int(end_year), inclusive="both")].copy()
    if work.empty:
        return pd.DataFrame()
    work["year"] = work["year"].astype(int)
    for col in [
        "social_sentiment_score",
        "social_sentiment_volatility",
        "social_sentiment_positive_share",
        "social_sentiment_negative_share",
        "social_sentiment_volume",
        "social_sentiment_platform_count",
        "social_sentiment_buzz",
    ]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        else:
            work[col] = np.nan
    if "social_sentiment_source" not in work.columns:
        work["social_sentiment_source"] = "missing"
    work["social_sentiment_source"] = work["social_sentiment_source"].fillna("missing").astype(str)
    meta = cities[["city_id", "city_name", "country", "continent"]].copy()
    meta["city_id"] = meta["city_id"].astype(str)
    work = work.merge(meta, on="city_id", how="left", suffixes=("", "_catalog"))
    for col in ["city_name", "country", "continent"]:
        alt = f"{col}_catalog"
        if alt in work.columns:
            work[col] = work.get(col).fillna(work[alt]) if col in work.columns else work[alt]
            work = work.drop(columns=[alt], errors="ignore")
    keep = [
        "city_id",
        "city_name",
        "country",
        "continent",
        "year",
        "social_sentiment_score",
        "social_sentiment_volatility",
        "social_sentiment_positive_share",
        "social_sentiment_negative_share",
        "social_sentiment_volume",
        "social_sentiment_platform_count",
        "social_sentiment_buzz",
        "social_sentiment_source",
    ]
    return work[keep].drop_duplicates(["city_id", "year"], keep="last").sort_values(["city_id", "year"])


def build_city_social_sentiment_yearly(
    *,
    max_cities: int = 295,
    start_year: int = 2015,
    end_year: int = 2025,
    use_cache: bool = True,
    max_records: int = 80,
    request_timeout: int = 25,
    sleep_seconds: float = 0.15,
    social_domains: Sequence[str] = DEFAULT_SOCIAL_DOMAINS,
    progress_every: int = 2,
) -> Dict[str, Any]:
    """Crawl city social-discourse proxy and build city-year sentiment panel."""
    yearly_path = DATA_RAW / "city_social_sentiment_yearly.csv"
    posts_path = DATA_RAW / "social_sentiment_posts.csv"

    cities = load_city_catalog(max_cities=int(max_cities))
    city_rows = cities[["city_id", "city_name", "country", "continent"]].copy()
    years_all = list(range(int(start_year), int(end_year) + 1))
    full_pairs = {
        (str(city_id), int(year))
        for city_id in city_rows["city_id"].astype(str).tolist()
        for year in years_all
    }

    cached_yearly = pd.DataFrame()
    if use_cache and yearly_path.exists():
        try:
            cached_yearly = _standardize_cached_social_yearly(
                pd.read_csv(yearly_path),
                cities,
                start_year=int(start_year),
                end_year=int(end_year),
            )
        except Exception:  # noqa: BLE001
            cached_yearly = pd.DataFrame()

    done_pairs: set[tuple[str, int]] = set()
    if not cached_yearly.empty:
        src = cached_yearly["social_sentiment_source"].fillna("missing").astype(str).str.lower()
        done_mask = src.isin(
            {
                "gdelt_social_media_proxy",
                "gdelt_city_news_proxy",
                "gdelt_no_hits",
                "external_social_posts",
                "external_social_city_year",
            }
        )
        done_pairs = {
            (str(row.city_id), int(row.year))
            for row in cached_yearly.loc[done_mask, ["city_id", "year"]].itertuples(index=False)
        }
        if done_pairs == full_pairs:
            years = pd.to_numeric(cached_yearly["year"], errors="coerce").dropna().astype(int)
            summary = {
                "status": "ok_cached",
                "rows": int(len(cached_yearly)),
                "cities": int(cached_yearly["city_id"].astype(str).nunique()),
                "year_range": [int(years.min()), int(years.max())] if not years.empty else None,
                "path": str(yearly_path),
            }
            dump_json(DATA_PROCESSED / "social_sentiment_summary.json", summary)
            return summary

    todo_pairs = full_pairs - done_pairs

    post_rows: List[dict] = []
    errors: List[dict] = []
    total_queries = 0
    no_hit_pairs = 0

    for city_idx, city in enumerate(city_rows.itertuples(index=False), start=1):
        c_name = str(city.city_name)
        city_years_pending = [
            int(year)
            for year in years_all
            if (str(city.city_id), int(year)) in todo_pairs
        ]
        if city_years_pending:
            LOGGER.info(
                "GDELT city %s/%s start: %s pending_years=%s",
                int(city_idx),
                int(len(city_rows)),
                c_name,
                int(len(city_years_pending)),
            )
        for year in years_all:
            if (str(city.city_id), int(year)) not in todo_pairs:
                continue
            query = ""
            query_source = "gdelt_no_hits"
            articles: List[dict] = []
            for query_source, query in _gdelt_queries_for_city(c_name, str(city.country), social_domains):
                total_queries += 1
                try:
                    articles = _fetch_gdelt_articles(
                        query,
                        start_dt=_gdelt_datetime(year, is_end=False),
                        end_dt=_gdelt_datetime(year, is_end=True),
                        max_records=int(max_records),
                        timeout=int(request_timeout),
                    )
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "city_id": str(city.city_id),
                            "city_name": c_name,
                            "year": int(year),
                            "query_source": str(query_source),
                            "error": str(exc),
                        }
                    )
                    articles = []
                if articles:
                    break

            if not articles:
                no_hit_pairs += 1
                if total_queries % max(1, int(progress_every)) == 0:
                    LOGGER.info(
                        "GDELT progress: queries=%s posts=%s no_hits=%s errors=%s",
                        int(total_queries),
                        int(len(post_rows)),
                        int(no_hit_pairs),
                        int(len(errors)),
                    )
                time.sleep(max(0.0, float(sleep_seconds)))
                continue

            for art in articles:
                title = str(art.get("title", "") or "")
                source = str(art.get("source", "") or "")
                domain = str(art.get("domain", "") or "")
                url = str(art.get("url", "") or "")
                seendate = art.get("seendate")
                sent = _score_gdelt_article(art)
                post_rows.append(
                    {
                        "city_id": str(city.city_id),
                        "city_name": c_name,
                        "country": str(city.country),
                        "continent": str(city.continent),
                        "year": int(year),
                        "title": title,
                        "domain": domain,
                        "source": source,
                        "url": url,
                        "seendate": seendate,
                        "platform": domain if domain else source,
                        "sentiment_score": float(sent),
                        "social_sentiment_source": str(query_source),
                        "tone": art.get("tone"),
                        "query": query,
                    }
                )

            if total_queries % max(1, int(progress_every)) == 0:
                LOGGER.info(
                    "GDELT progress: queries=%s posts=%s no_hits=%s errors=%s",
                    int(total_queries),
                    int(len(post_rows)),
                    int(no_hit_pairs),
                    int(len(errors)),
                )
            time.sleep(max(0.0, float(sleep_seconds)))

    posts_df = pd.DataFrame(post_rows)
    if posts_df.empty and not todo_pairs:
        years = pd.to_numeric(cached_yearly["year"], errors="coerce").dropna().astype(int)
        summary = {
            "status": "ok_cached",
            "rows": int(len(cached_yearly)),
            "cities": int(cached_yearly["city_id"].astype(str).nunique()) if not cached_yearly.empty else 0,
            "year_range": [int(years.min()), int(years.max())] if not years.empty else None,
            "path": str(yearly_path),
        }
        dump_json(DATA_PROCESSED / "social_sentiment_summary.json", summary)
        return summary

    if posts_df.empty and cached_yearly.empty:
        summary = {
            "status": "failed",
            "reason": "no_social_posts_collected",
            "queries": int(total_queries),
            "errors": int(len(errors)),
            "error_sample": errors[:20],
        }
        dump_json(DATA_PROCESSED / "social_sentiment_summary.json", summary)
        return summary

    agg = pd.DataFrame()
    std_posts = pd.DataFrame()
    if not posts_df.empty:
        agg, std_posts = aggregate_social_posts(
            posts_df,
            cities,
            start_year=int(start_year),
            end_year=int(end_year),
            source_label="gdelt_social_media_proxy",
        )

    if agg.empty and cached_yearly.empty and todo_pairs:
        summary = {
            "status": "failed",
            "reason": "aggregation_empty",
            "queries": int(total_queries),
            "post_rows": int(len(posts_df)),
            "errors": int(len(errors)),
            "error_sample": errors[:20],
        }
        dump_json(DATA_PROCESSED / "social_sentiment_summary.json", summary)
        return summary

    new_pairs_df = pd.DataFrame(sorted(todo_pairs), columns=["city_id", "year"]) if todo_pairs else pd.DataFrame(columns=["city_id", "year"])
    if not new_pairs_df.empty:
        yearly_new = new_pairs_df.merge(agg, on=["city_id", "year"], how="left")
        yearly_new["social_sentiment_volume"] = pd.to_numeric(yearly_new["social_sentiment_volume"], errors="coerce").fillna(0.0)
        yearly_new["social_sentiment_platform_count"] = pd.to_numeric(
            yearly_new["social_sentiment_platform_count"], errors="coerce"
        ).fillna(0.0)
        yearly_new["social_sentiment_buzz"] = pd.to_numeric(yearly_new["social_sentiment_buzz"], errors="coerce").fillna(0.0)
        yearly_new["social_sentiment_source"] = yearly_new["social_sentiment_source"].fillna("gdelt_no_hits").astype(str)
    else:
        yearly_new = pd.DataFrame(columns=["city_id", "year"])

    meta_cols = cities[["city_id", "city_name", "country", "continent"]].copy()
    meta_cols["city_id"] = meta_cols["city_id"].astype(str)
    yearly_new = yearly_new.merge(meta_cols, on="city_id", how="left")
    yearly = pd.concat([cached_yearly, yearly_new], ignore_index=True)
    yearly = yearly.drop_duplicates(["city_id", "year"], keep="last")
    yearly = yearly.sort_values(["city_id", "year"])
    yearly = yearly[
        [
            "city_id",
            "city_name",
            "country",
            "continent",
            "year",
            "social_sentiment_score",
            "social_sentiment_volatility",
            "social_sentiment_positive_share",
            "social_sentiment_negative_share",
            "social_sentiment_volume",
            "social_sentiment_platform_count",
            "social_sentiment_buzz",
            "social_sentiment_source",
        ]
    ].copy()

    existing_posts = pd.DataFrame()
    if use_cache and posts_path.exists():
        try:
            existing_posts = pd.read_csv(posts_path)
        except Exception:  # noqa: BLE001
            existing_posts = pd.DataFrame()
    posts_out = pd.concat([existing_posts, std_posts], ignore_index=True) if not existing_posts.empty else std_posts
    if not posts_out.empty:
        dedupe_cols = [c for c in ["city_id", "year", "url", "title", "platform"] if c in posts_out.columns]
        if dedupe_cols:
            posts_out = posts_out.drop_duplicates(dedupe_cols, keep="last")
        posts_out.to_csv(posts_path, index=False)
    yearly.to_csv(yearly_path, index=False)

    summary = {
        "status": "ok",
        "queries": int(total_queries),
        "cities": int(yearly["city_id"].nunique()),
        "rows": int(len(yearly)),
        "post_rows": int(len(posts_out)) if 'posts_out' in locals() else int(len(std_posts)),
        "coverage_ratio": float((pd.to_numeric(yearly["social_sentiment_volume"], errors="coerce") > 0).mean()),
        "year_range": [int(start_year), int(end_year)],
        "error_count": int(len(errors)),
        "no_hit_pairs": int(no_hit_pairs),
        "error_sample": errors[:30],
        "cached_pairs_used": int(len(done_pairs)),
        "queried_pairs": int(len(todo_pairs)),
        "paths": {
            "posts": str(posts_path),
            "yearly": str(yearly_path),
        },
    }
    dump_json(DATA_PROCESSED / "social_sentiment_summary.json", summary)
    if errors:
        err_path = DATA_RAW / "social_sentiment_crawl_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False)
        summary["paths"]["errors"] = str(err_path)
    return summary


__all__ = [
    "simple_sentiment_score",
    "aggregate_social_posts",
    "build_city_social_sentiment_yearly",
    "DEFAULT_SOCIAL_DOMAINS",
]
