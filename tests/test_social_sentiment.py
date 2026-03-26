from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

TEST_OUTPUT_ROOT = Path(tempfile.gettempdir()) / "urban_pulse_test_artifacts"
os.environ["URBAN_PULSE_OUTPUT_ROOT"] = str(TEST_OUTPUT_ROOT)

from src.social_sentiment import aggregate_social_posts, simple_sentiment_score


class SocialSentimentTests(unittest.TestCase):
    def test_simple_sentiment_score_direction(self) -> None:
        pos = simple_sentiment_score("great growth and innovation with vibrant streets")
        neg = simple_sentiment_score("crisis, decline, unemployment and pollution")
        self.assertGreater(pos, 0.0)
        self.assertLess(neg, 0.0)

    def test_aggregate_social_posts_city_year(self) -> None:
        cities = pd.DataFrame(
            [
                {"city_id": "city_a", "city_name": "Alpha"},
                {"city_id": "city_b", "city_name": "Beta"},
            ]
        )
        posts = pd.DataFrame(
            [
                {"city_id": "city_a", "year": 2024, "text": "great growth and safe city", "platform": "reddit"},
                {"city_id": "city_a", "year": 2024, "text": "pollution and decline", "platform": "x"},
                {"city_id": "city_b", "year": 2024, "text": "stable and vibrant innovation", "platform": "reddit"},
            ]
        )

        agg, post_std = aggregate_social_posts(
            posts,
            cities,
            start_year=2024,
            end_year=2024,
            source_label="unit_test",
        )

        self.assertFalse(agg.empty)
        self.assertEqual(int(len(post_std)), 3)
        self.assertTrue({"city_id", "year", "social_sentiment_score", "social_sentiment_volume"}.issubset(agg.columns))

        row_a = agg[(agg["city_id"] == "city_a") & (agg["year"] == 2024)]
        self.assertEqual(int(row_a["social_sentiment_volume"].iloc[0]), 2)
        self.assertEqual(str(row_a["social_sentiment_source"].iloc[0]), "unit_test")


if __name__ == "__main__":
    unittest.main()
