| variable | role | resolution | city_observed | coverage_share | main_prediction | main_causal | validation_only | context_only | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| log_viirs_ntl | Primary slow outcome | city-year polygon satellite | yes | 1 | yes | yes | no | no | Annual VIIRS, full 2015-2025 panel coverage. |
| no2_trop_anomaly_mean | Primary fast outcome | city-year polygon satellite | yes | 0.727 | no | yes | no | no | Observed 2018-2025 only; backcasted version enters benchmark modules only. |
| physical_built_expansion_primary | Slow structural proxy | city-year polygon satellite | yes | 1 | no | no | no | no | Used in mechanism decomposition and appendix proxy checks. |
| road_growth_intensity | Local transport/friction proxy | city-year OSM-derived | yes | 1 | yes | no | no | no | Used in dynamic features and supplementary mechanism proxy tests. |
| poi_total_yoy | Middle-speed activity proxy | city-year OSM-derived | yes | 0.909 | yes | no | no | no | Supplementary activity proxy; coverage starts after first within-city year. |
| temperature_mean + precipitation_sum | Climate controls | city-year weather | yes | 1 | yes | yes | no | yes | Observed city-year controls, not interpreted as direct growth outcomes. |
| baseline_population_log | Baseline scale control | mixed / baseline-derived | mixed | 1 | yes | yes | no | yes | Derived baseline control; retained for adjustment, not as a city-observed outcome. |
| knowledge_capital_raw | Innovation-side mechanism proxy | mixed / macro-derived composite | no | 1 | yes | no | no | no | Used in theory fit and mechanism decomposition, not as the primary policy outcome. |
| country-year macro context block | Contextual macro controls | country-year | no | 1 | yes | yes | no | yes | Retained as background context only; never interpreted as direct city-level measurement. |
| gdp_total_ppp_observed | External validation anchor | city/metro-year observed | partial | 0.231 | no | no | yes | no | Used for calibration and future-slowdown validation in the observed-GDP subset. |
| gdp_total_local_observed | Observed GDP robustness anchor | city/metro-year observed | partial | 0.09 | no | no | yes | no | Local-currency robustness anchor; used as supplementary external evidence only. |
