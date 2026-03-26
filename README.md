# MACRO-City Engine

Multi-modal Analytics and Causal Research Observatory for Cities

兼容性说明：仓库内部仍保留一部分 `urban_pulse_*` 文件名、环境变量和导出前缀，以避免打断既有 pipeline；公开展示名称统一为 `MACRO-City Engine`。

基于全球城市面板数据的城市发展多维评估系统，覆盖：
- 空间锚点：城市目录新增 `geometry_wkt`，优先采用 GHSL Urban Centres，多边形缺口按 GHSL FUA / OECD cities 回填；遥感裁剪基于 polygon，而非 point-buffer
- 数据采集：World Bank + Open-Meteo + OSM（严格模式默认先尝试 `verified source` 过滤；当 verified 覆盖不足时自动回退到 `objective source` 全样本，并在审计文件中明确标记 `filter_basis=objective_city_complete_fallback`）
- 扩展数据：主面板并入 8 个 World Bank 扩展指标（专利、科研人员、高技术出口、就业率、城市化率、电力可及、固定宽带、PM2.5），并输出 `extra_wb_source` 审计字段
- 物理观测：支持接入 GHSL 年度建成区面积/体积表，`ghsl_built_surface_km2` 作为“物理建成环境扩张”的主干真值变量
- 宏观降尺度：国家 GDP 仅按 `viirs_ntl_sum` 的国家内占比分配到城市，生成 `gdp_disaggregated_by_ntl`；OSM 道路、建筑、POI 不参与经济指标降尺度
- 众包测绘审计：当 `osm_hist_building_count_yoy` 或 `poi_total_yoy` 出现单年异常跳变时生成 `is_mapping_surge`
- 建模：多目标城市指标预测（线性模型 + 树模型 + 图扩散特征基线，严格采用 t->t+1 预测协议）
- 经济学分析：双向固定效应 DID、DML-DID（双重机器学习）、异质性 DID（分洲/分收入组 + 动态相位异质性 + 相位规则敏感性）、事件研究、矩阵补全反事实（含placebo）、Beta 收敛、弹性回归、合成控制；主处理入口收敛到 `treated_city_direct_core`，主回归只允许稀疏外生控制并按 `iso3` 聚类标准误
- 政策处理：先用外部直接事件（WB Projects）构建处理变量，再用 WB 指标结构突变事件（`objective_indicator`）补齐缺口国家，之后才退化到客观宏观规则与 AI 推断；主 DID 口径仅使用 `direct_core`
- 情绪模块：`social_sentiment_*` 不进入主面板 PCA/主因果链，只保留 `has_sentiment` 掩码供子样本机制分析
- AI升级：城市表征学习（embedding）、脉搏状态概率建模、Pulse AI 动态引擎（加速/失速风险/城市类型）、Trajectory Regime AI（DTW+KMedoids 动态体制识别 + 转移特征 + Regime网络特征 + 候选模型自动择优 + 跨洲先验校准）、动态六层结构（Kinetic State + Transition Hazard + Graph Diffusion + Main Risk Fusion + Global Cycle + Policy/Event Lab）、状态门控融合（state-gated hazard fusion）、动态状态事件研究（entry event study with de-shocked relative effects）、跨洲同步网络（lead-lag sync network + permutation显著性）、政策实验室（counterfactual policy lab）、自适应象限阈值与1年迁移轨迹分析、冲击-脉搏响应 IRF 分析、跨洲泛化评估（含 Logit 基线对照、bootstrap 显著性区间）、分位无分布不确定性量化（Split-Conformal + Bootstrap区间）、动态融合显著性评估（AUC/Brier bootstrap CI + p值）、多期失速风险预测（1Y/2Y/3Y Markov-Regime Forecast）、时空因果模型（Causal-ST v2，含消融与政策仿真）、跨时空OOD基准评测（t->t+1）、解释一致性诊断（Permutation vs SHAP/Proxy）与跨年份特征漂移分析
- 可视化：Flask + Leaflet + Chart.js 全球仪表盘

## 目录结构

- `src/global_data.py`：全球数据采集、特征工程、指标构建
- `src/modeling.py`：预测模型训练与评估
- `src/econometrics.py`：计量经济学方法实现
- `src/provenance.py`：客观来源审计与严格样本过滤
- `src/representation.py`：城市embedding构建
- `src/pulse_state.py`：城市脉搏状态概率与转移矩阵
- `src/pulse_ai.py`：脉搏AI动态引擎（加速评分、失速风险、城市类型）
- `src/causal_st.py`：时空因果基线模型
- `src/benchmark_eval.py`：时序/空间OOD评测协议
- `src/ai_explainability.py`：AI解释一致性与特征漂移诊断
- `src/inference_reporting.py`：统一不确定性报告（CI/Bootstrap/多重检验）
- `src/dynamic_causal_envelope.py`：动态因果包络（事件包络/城市包络/洲际时序）
- `src/pulse_dynamics.py`：动态状态转移/生存风险/跨期预警诊断
- `src/pulse_nowcast.py`：动态现在预测（nowcast）与预测区间
- `src/dynamic_method_core.py`：主方法-强基线滚动评测、配对显著性与消融
- `src/export_dashboard.py`：可视化数据导出
- `src/pipeline.py`：全流程编排
- `run_pipeline.py`：一键执行入口
- `run_data_crawler.py`：统一多源真实数据爬取入口
- `run_social_crawler.py`：城市舆情情绪抓取入口
- `run_web.py`：仪表盘启动入口
- `run_realtime_monitor.py`：实时监测快照刷新入口（轻量）

## 快速开始

1. 运行全流程（默认严格真实数据模式，不允许合成降级，且默认启用 `verified source` 过滤）

```bash
python3 run_pipeline.py
```

如需强制使用“真实政策事件库”（缺失即失败，不允许规则回退）：

```bash
python3 run_pipeline.py --require-policy-events
```

如需在事件库缺失时自动抓取（World Bank Projects API）：

```bash
python3 run_pipeline.py --auto-build-policy-events
```

说明：默认会使用 `config.json` 的 `max_cities_default`（当前为 295）。当设置 `--max-cities` 时，系统会按“分洲均衡 + 国家轮转”抽样，不再按文件顺序截断。

可选：调整严格来源门槛（默认即为论文主结果推荐设置）：

```bash
python3 run_pipeline.py \
  --min-verified-city-retention 0.55 \
  --min-external-direct-share 0.70 \
  --max-ai-inferred-share 0.30
```

仅在探索阶段可放宽为“objective但非verified”的来源过滤（不建议用于最终结论）：

```bash
python3 run_pipeline.py --allow-non-verified-sources
```

严格模式下默认会为缺失城市尝试实时 OSM 抓取。若你处于离线环境，可显式关闭该抓取（会降低 verified 覆盖率）：

```bash
export URBAN_PULSE_STRICT_SKIP_LIVE_POI=1
```

严格模式下天气采集会在连续失败后触发“真实天气池客观插补”。可用下列参数调低触发阈值（离线环境建议 3）：

```bash
export URBAN_PULSE_STRICT_WEATHER_CIRCUIT=3
```

若你明确希望允许降级（不推荐研究场景）：

```bash
python3 run_pipeline.py --allow-synthetic-fallback
```

可选：将输出重定向到独立目录（避免覆盖主产物）：

```bash
export URBAN_PULSE_OUTPUT_ROOT=/tmp/urban_pulse_runs/main_run
```

可选：启用“真实政策事件表”处理变量。将 `data/raw/policy_events_registry_template.csv` 复制为
`data/raw/policy_events_registry.csv` 并填入审计字段（`iso3/start_year/source_ref` 等）。若未提供，系统自动回退到规则处理变量并在 `data/processed/policy_design.json` 记录。

可选：启用“路网分层 + VIIRS 夜光”增强（推荐）。可直接通过爬虫自动构建，或手工准备文件后运行 `run_pipeline.py`。

自动构建（Overpass 路网快照 + NOAA VIIRS 夜光抽样）：

```bash
python3 run_data_crawler.py \
  --max-cities 295 --start-year 2015 --end-year 2025 \
  --skip-macro --skip-extra-wb --skip-policy --skip-weather --skip-poi \
  --crawl-road --crawl-viirs --road-radius-m 2000
```

说明：

- 路网为 OSM 实测快照，并扩展为年度面板用于分层诊断。
- VIIRS 来自 NOAA Nightly Radiance 服务，当前可获得的在线时段有限（以实际服务目录为准）。

可选：补充 OSM 历史时序（ohsome API），生成 `data/raw/city_osm_history_yearly.csv`：

```bash
python3 run_data_crawler.py \
  --max-cities 295 --start-year 2015 --end-year 2025 \
  --skip-macro --skip-extra-wb --skip-policy --skip-weather --skip-poi \
  --crawl-osm-history
```

可选：补充“城市舆论氛围”指标（社媒讨论情绪量化）：

```bash
python3 run_data_crawler.py \
  --max-cities 295 --start-year 2015 --end-year 2025 \
  --skip-macro --skip-extra-wb --skip-policy --skip-weather --skip-poi \
  --crawl-social-sentiment --social-max-records 80
```

也可独立运行舆情抓取：

```bash
python3 run_social_crawler.py --max-cities 295 --start-year 2015 --end-year 2025 --max-records 80
```

手工准备时，文件格式至少满足：

- `data/raw/city_road_network_yearly.csv`：`city_id,year,road_length_km_total,arterial_share,intersection_density`
- `data/raw/viirs_city_monthly.csv`：`city_id,year,month,radiance`（可选 `lit_area_km2`）
- `data/raw/city_social_sentiment_yearly.csv`：`city_id,year,social_sentiment_score,...`（可选）
- `data/raw/social_sentiment_posts.csv`：`city_id,year,text/sentiment_score,...`（可选）

模板文件已提供：

- `data/raw/city_road_network_yearly_template.csv`
- `data/raw/viirs_city_monthly_template.csv`
- `data/raw/city_social_sentiment_yearly_template.csv`
- `data/raw/social_sentiment_posts_template.csv`

也可独立执行爬取脚本自动生成事件表：

```bash
python3 run_policy_crawler.py --max-cities 295 --start-year 2015 --end-year 2025
```

如需一键抓取多源数据（宏观 + 扩展WB指标 + 政策事件 + 天气 + POI）：

```bash
python3 run_data_crawler.py --max-cities 295 --start-year 2015 --end-year 2025 --strict-weather-circuit 3 --strict-skip-live-poi
```

若网络波动较大、需要提升严格模式下 POI 真实抓取完成率，可调节严格参数：

```bash
python3 run_data_crawler.py \
  --max-cities 295 --start-year 2015 --end-year 2025 \
  --skip-macro --skip-extra-wb --skip-policy --skip-weather \
  --strict-poi-timeout 9 --strict-poi-retries 1 --strict-poi-backoff 0.7 \
  --strict-poi-sleep 0.15 --strict-poi-max-consec-fail 20 --strict-poi-max-total-fail 80
```

离线/半离线场景下可只抓取可达源：

```bash
python3 run_data_crawler.py --skip-weather --skip-poi --extra-wb-cache-only
```

2. 启动可视化

```bash
python3 run_web.py
```

访问：
- `http://127.0.0.1:8000/`（统一单页工作台，默认入口）

兼容旧链接（会进入统一单页并切到对应标签）：
- `http://127.0.0.1:8000/dashboard/dynamics`
- `http://127.0.0.1:8000/dashboard/method-core`
- `http://127.0.0.1:8000/dashboard/realtime`
- `http://127.0.0.1:8000/dashboard/identification`
- `http://127.0.0.1:8000/dashboard/external-validity`
- `http://127.0.0.1:8000/dashboard/policy-rl`
- `http://127.0.0.1:8000/dashboard/top-tier`
- `http://127.0.0.1:8000/dashboard/top-tier-story`
- `http://127.0.0.1:8000/dashboard/full`

3. 仅刷新实时监测快照（不重跑全流程）

```bash
python3 run_realtime_monitor.py
```

定时刷新（例如每 600 秒一次）：

```bash
python3 run_realtime_monitor.py --loop-seconds 600
```

默认会启动后台实时监测线程（按固定周期刷新 nowcast 快照），可通过环境变量调整：

```bash
export URBAN_PULSE_MONITOR_REFRESH_SECONDS=900
```

如需禁用后台线程（仅按需触发）：

```bash
export URBAN_PULSE_DISABLE_BACKGROUND_MONITOR=1
```

实时监测 API：
- `GET /api/realtime/status`
- `GET /api/realtime/countries`
- `GET /api/realtime/alerts`
- `GET /api/realtime/continents`
- `GET /api/realtime/sentinel`
- `GET /api/realtime/stream`（SSE 实时推送）
- `POST /api/realtime/trigger`（手动触发更新）

Pulse AI 动态策略 API：
- `GET /api/pulse_ai_dynamic_policy`
- `GET /api/pulse_ai_dynamic_policy_rl`
- `GET /api/pulse_ai_dynamic_policy_rl_city`
- `GET /api/pulse_ai_dynamic_policy_rl_state`
- `GET /api/pulse_ai_dynamic_policy_rl_ope`
- `GET /api/pulse_ai_dynamic_policy_rl_ablation`
- `GET /api/pulse_ai_dynamic_policy_rl_continent_ope`
- `GET /api/pulse_ai_dynamic_policy_rl_continent_action`
- `GET /api/pulse_ai_dynamic_index_latest`
- `GET /api/pulse_ai_dynamic_index_continent`

Trajectory Regime 动态 API：
- `GET /api/pulse_ai_regime_share`
- `GET /api/pulse_ai_regime_dynamics`
- `GET /api/pulse_ai_regime_transition`
- `GET /api/pulse_ai_trajectory_regimes`

Identification Stress-Test API：
- `GET /api/idplus_summary`
- `GET /api/idplus_stress_index`
- `GET /api/idplus_pretrend_geometry`
- `GET /api/idplus_concordance_pairs`
- `GET /api/idplus_leave_continent_stability`
- `GET /api/econometric_policy_source_sensitivity`
- `GET /api/econometric_source_event_study_points`

Dynamic Causal Envelope API：
- `GET /api/dce_summary`
- `GET /api/dce_event`
- `GET /api/dce_event_bootstrap`
- `GET /api/dce_city_scores`
- `GET /api/dce_continent_year`
- `GET /api/dce_regime_summary`
- `GET /api/dce_continent_stability`

Pulse Dynamics API：
- `GET /api/pulse_dynamics_summary`
- `GET /api/pulse_dynamics_transition`
- `GET /api/pulse_dynamics_spell_hazard`
- `GET /api/pulse_dynamics_resilience`
- `GET /api/pulse_dynamics_warning`

Pulse Nowcast API：
- `GET /api/pulse_nowcast_summary`
- `GET /api/pulse_nowcast_latest`
- `GET /api/pulse_nowcast_history`
- `GET /api/pulse_nowcast_global`

Dynamic Method Core API：
- `GET /api/dynamic_method_core_summary`
- `GET /api/dynamic_method_core_metrics`
- `GET /api/dynamic_method_core_significance`
- `GET /api/dynamic_method_core_ablation`

External Validity API：
- `GET /api/external_validity`
- `GET /api/external_validity_indicators`
- `GET /api/external_validity_rank_corr`

Top-tier 审稿视角 API：
- `GET /api/top_tier`
- `GET /api/top_tier_gate_checks`
- `GET /api/top_tier_evidence_convergence`
- `GET /api/top_tier_identification_spectrum`
- `GET /api/top_tier_innovation_frontier`
- `GET /api/top_tier_story_bundle`
- `GET /api/top_tier_story_markdown`
  - 两个 story 接口支持 `?continent=<name>&year=<yyyy>`，用于按筛选导出评审叙事快照
- `GET /api/frontend_bundle`
  - 统一单页前端聚合接口，支持 `?continent=<name>&year=<yyyy>`

投稿就绪状态 API：
- `GET /api/submission_readiness`

## 主要产物

- 面板数据（原始整合）：`data/processed/global_city_panel.csv`
- 面板数据（严格来源过滤后建模样本）：`data/processed/global_city_panel_strict.csv`
- 建模评估：`data/outputs/model_metrics.json`
  - AI增益诊断（结构特征 vs 工程特征 vs Pulse动态特征）：`data/outputs/model_ai_incrementality.json`
  - AI增益明细表：`data/outputs/model_ai_incrementality.csv`
- 计量结果：`data/outputs/econometric_summary.json`
  - 统一不确定性报告：`data/outputs/inference_reporting_summary.json`
  - 主结果 CI 表：`data/outputs/inference_main_results.csv`
  - 多重检验校正表（BH/Bonferroni）：`data/outputs/inference_multiple_testing.csv`
  - 外部效度桥接：`data/outputs/external_validity_summary.json`
  - 外部效度明细：`data/outputs/external_validity_indicator_results.csv`
  - 外部效度年度秩相关：`data/outputs/external_validity_rank_corr_by_year.csv`
  - 政策来源口径敏感性总表：`data/outputs/econometric_policy_source_sensitivity.csv`
  - Stacked lead-placebo（cohort-aware, event-window）结果：`did_stacked_lead_placebo` 与 `data/outputs/did_stacked_lead_placebo_main_composite_index.csv`
  - 动态相位异质性 DID（AI 相图分层 + 交互项识别）：`dynamic_phase_heterogeneity` 与 `data/outputs/dynamic_phase_heterogeneity_composite_index.csv`
  - 动态相位规则敏感性（多规则阈值、交互项符号一致性）：`dynamic_phase_rule_sensitivity` 与 `data/outputs/dynamic_phase_rule_sensitivity_composite_index.csv`
  - 政策来源分口径事件研究点表：`data/outputs/econometric_source_event_study_points.csv`
  - 动态机制分解表（经济活力/宜居/创新通道）：`data/outputs/mechanism_decomposition_main_did_treatment.csv`
  - 动态主设计可选峰值冲击口径：`main_design_variant=intense_external_peak`
  - 政策来源敏感性采用“共同动态冲击锚点 + 来源强度通道”比较模式：`source_event_design_robustness.comparison_mode=common_dynamic_shock_anchor`
  - 含政策参考年份自动识别：`policy_reference_year`
  - 含匹配稳健性：`did_matched_trend`
  - 含分期稳健性：`staggered_did`
  - 含 not-yet-treated 稳健性：`not_yet_treated_did`
- 客观来源审计：
  - `data/processed/source_audit_summary.json`
  - `data/processed/source_audit_city.csv`
  - `data/processed/source_audit_combo.csv`
  - `data/processed/coverage_summary.json`
  - `data/processed/coverage_by_country.csv`
  - `data/processed/coverage_by_continent.csv`
  - `data/processed/policy_event_registry_audit.json`
  - `data/processed/policy_event_registry_coverage_by_continent.csv`
  - `data/processed/policy_event_objective_indicator_summary.json`
  - `data/processed/policy_event_objective_macro_summary.json`
  - 注：`source_audit_summary.json` 中 `filter_basis`、`verified_row_ratio`、`verified_complete_city_ratio`、`city_retention_ratio` 为严格来源可核验性的核心指标；`data_quality_summary.json` 中 `objective_row_ratio` 与 `direct_verified_row_ratio` 用于区分“客观可审计”与“直接可核验”两层口径。
- AI动态模块：
  - `data/outputs/city_embeddings.csv`
  - `data/outputs/pulse_state_probabilities.csv`
  - `data/outputs/pulse_transition_matrix.csv`
  - `data/outputs/pulse_state_summary.json`
  - `data/outputs/pulse_ai_scores.csv`
  - `data/outputs/pulse_ai_city_latest.csv`
  - `data/outputs/pulse_ai_archetypes.csv`
  - `data/outputs/pulse_ai_trajectory_regimes.csv`
  - `data/outputs/pulse_ai_regime_by_year.csv`
  - `data/outputs/pulse_ai_regime_year_share.csv`
  - `data/outputs/pulse_ai_regime_transition_matrix.csv`
  - `data/outputs/pulse_ai_regime_medoids.csv`
  - `data/outputs/pulse_ai_cross_continent_generalization.csv`
  - `data/outputs/pulse_ai_shock_years.csv`
  - `data/outputs/pulse_ai_shock_irf_regime.csv`
  - `data/outputs/pulse_ai_shock_irf_vulnerability.csv`
  - `data/outputs/pulse_ai_horizon_forecast.csv`
  - `data/outputs/pulse_ai_dynamic_hazard_latest.csv`
  - `data/outputs/pulse_ai_dynamic_graph_latest.csv`
  - `data/outputs/pulse_ai_dynamic_graph_curve.csv`
  - `data/outputs/pulse_ai_dynamic_critical_latest.csv`
  - `data/outputs/pulse_ai_dynamic_critical_decile.csv`
  - `data/outputs/pulse_ai_dynamic_gate_weights.csv`
  - `data/outputs/pulse_ai_dynamic_main_fusion_curve.csv`
  - `data/outputs/pulse_ai_dynamic_main_fusion_pareto.csv`
  - `data/outputs/pulse_ai_dynamic_main_fusion_state_gate.csv`
  - `data/outputs/pulse_ai_dynamic_main_fusion_continent_eval.csv`
  - `data/outputs/pulse_ai_dynamic_main_fusion_continent_robust_curve.csv`
  - `data/outputs/pulse_ai_dynamic_main_fusion_continent_adaptive_weights.csv`
  - `data/outputs/pulse_ai_dynamic_phase_field.csv`
  - `data/outputs/pulse_ai_dynamic_phase_latest.csv`
  - `data/outputs/pulse_ai_dynamic_cycle.csv`
  - `data/outputs/pulse_ai_dynamic_state_event_study.csv`
  - `data/outputs/pulse_ai_dynamic_state_event_panel.csv`
  - `data/outputs/pulse_ai_dynamic_sync_network.csv`
  - `data/outputs/pulse_ai_dynamic_policy_lab.csv`
  - `data/outputs/pulse_ai_dynamic_policy_lab_summary.csv`
  - `data/outputs/pulse_ai_dynamic_policy_rl_city.csv`
  - `data/outputs/pulse_ai_dynamic_policy_rl_action_summary.csv`
  - `data/outputs/pulse_ai_dynamic_policy_rl_state_value.csv`
  - `data/outputs/pulse_ai_dynamic_policy_rl_ope.csv`
  - `data/outputs/pulse_ai_dynamic_policy_rl_ablation.csv`
  - `data/outputs/pulse_ai_dynamic_policy_rl_continent_ope.csv`
  - `data/outputs/pulse_ai_dynamic_policy_rl_continent_action.csv`
  - `data/outputs/pulse_ai_dynamic_index_series.csv`
  - `data/outputs/pulse_ai_dynamic_index_latest.csv`
  - `data/outputs/pulse_ai_dynamic_index_continent_year.csv`
  - `data/outputs/dynamic_causal_envelope_summary.json`
  - `data/outputs/dynamic_causal_envelope_event.csv`
  - `data/outputs/dynamic_causal_envelope_event_bootstrap.csv`
  - `data/outputs/dynamic_causal_envelope_regime.csv`
  - `data/outputs/dynamic_causal_envelope_regime_summary.csv`
  - `data/outputs/dynamic_causal_envelope_city_scores.csv`
  - `data/outputs/dynamic_causal_envelope_continent_year.csv`
  - `data/outputs/dynamic_causal_envelope_continent_stability.csv`
  - `data/outputs/pulse_dynamics_summary.json`
  - `data/outputs/pulse_dynamics_state_panel.csv`
  - `data/outputs/pulse_dynamics_state_thresholds.csv`
  - `data/outputs/pulse_dynamics_transition_tensor.csv`
  - `data/outputs/pulse_dynamics_stall_spells.csv`
  - `data/outputs/pulse_dynamics_accel_spells.csv`
  - `data/outputs/pulse_dynamics_spell_hazard.csv`
  - `data/outputs/pulse_dynamics_resilience_halflife.csv`
  - `data/outputs/pulse_dynamics_warning_horizon.csv`
  - `data/outputs/pulse_nowcast_summary.json`
  - `data/outputs/pulse_nowcast_continent_latest.csv`
  - `data/outputs/pulse_nowcast_continent_history.csv`
  - `data/outputs/pulse_nowcast_backtest_metrics.csv`
  - `data/outputs/pulse_nowcast_global.csv`
  - `data/outputs/dynamic_method_core_summary.json`
  - `data/outputs/dynamic_method_core_predictions.csv`
  - `data/outputs/dynamic_method_core_metrics.csv`
  - `data/outputs/dynamic_method_core_significance.csv`
  - `data/outputs/dynamic_method_core_ablation.csv`
  - `data/outputs/pulse_ai_summary.json`
  - `web/static/data/realtime_status.json`
  - `web/static/data/realtime_city_monitor.csv`
  - `web/static/data/realtime_country_monitor.csv`
  - `web/static/data/realtime_continent_monitor.csv`
  - `web/static/data/realtime_alerts.csv`
  - `web/static/data/realtime_sentinel.csv`
  - `data/outputs/realtime_monitor_history.jsonl`
  - `data/outputs/causal_st_summary.json`
  - `data/outputs/causal_st_counterfactual.csv`
  - `data/outputs/causal_st_dynamic_att.csv`
  - `data/outputs/causal_st_cate_continent.csv`
  - `data/outputs/causal_st_policy_simulation.csv`
  - `data/outputs/causal_st_ablation.json`
  - `data/outputs/causal_st_ablation.csv`
  - `data/outputs/benchmark_scores.json`
  - `data/outputs/experiment_enhancements.json`
    - 负控检验采用分方法汇总（`twfe_did` + `staggered_did`），主结论口径默认 `primary_method=staggered_did`
  - `data/outputs/dynamic_phase_heterogeneity_composite_index.csv`
  - `data/outputs/dynamic_phase_rule_sensitivity_composite_index.csv`
  - `data/outputs/experiment_temporal_split_sensitivity.csv`
  - `data/outputs/experiment_spatial_ood_dispersion.csv`
  - `data/outputs/experiment_continent_transfer_matrix.csv`
  - `data/outputs/experiment_continent_transfer_summary.csv`
  - `data/outputs/experiment_did_placebo.csv`
  - `data/outputs/experiment_did_permutation_distribution.csv`
  - `data/outputs/experiment_did_specification_curve.csv`
  - `data/outputs/experiment_did_negative_controls.csv`
  - `data/outputs/did_matched_pairs.csv`
  - `data/outputs/did_matched_balance.csv`
  - `data/outputs/staggered_cohort_assignments.csv`
  - `data/outputs/staggered_cohort_distribution.csv`
  - `data/outputs/staggered_att_by_cohort_year.csv`
  - `data/outputs/staggered_event_time_summary.csv`
  - `data/outputs/not_yet_treated_did_by_cohort.csv`
  - `data/outputs/experiment_leave_one_continent_out.csv`
  - `data/outputs/experiment_pulse_calibration_bins.csv`
  - `data/outputs/experiment_pulse_bootstrap_metrics.csv`
  - `data/outputs/experiment_pulse_topk.csv`
  - `data/outputs/experiment_pulse_group_fairness.csv`
  - `data/outputs/experiment_pulse_group_fairness_summary.csv`
  - `data/outputs/experiment_pulse_decision_curve.csv`
  - AI解释一致性：`data/outputs/ai_explainability_summary.json`
  - 特征解释一致性明细：`data/outputs/ai_explainability_consistency.csv`
  - 跨年份特征重要性：`data/outputs/ai_feature_importance_by_year.csv`
  - 跨年份特征漂移汇总：`data/outputs/ai_feature_importance_drift_summary.csv`
  - `data/outputs/external_validity_summary.json`
  - `data/outputs/external_validity_indicator_results.csv`
  - `data/outputs/external_validity_rank_corr_by_year.csv`
- 合成控制明细：
  - `data/outputs/synthetic_control_timeseries.csv`
  - `data/outputs/synthetic_control_weights.csv`
- 挑战杯报告草稿：`reports/final_report.md`
- 论文图表清单：`paper/figures_tables_guide.md`
- 投稿增强清单：`paper/submission_checklist.md`
- 投稿就绪执行报告：`paper/submission_readiness_report.md`
- 投稿就绪审计 JSON：`data/outputs/submission_readiness.json`
- 可复现说明：`paper/reproducibility.md`
- 锁定依赖：`requirements_lock.txt`
- Conda 锁环境：`paper/environment_lock.yml`
- LaTeX 投稿包：`paper/latex/urban_pulse_main.tex` + `paper/latex/references.bib`
- 论文图表自动导出脚本：`paper/scripts/generate_paper_assets.py`
- 论文扩展图表自动导出脚本：`paper/scripts/generate_extended_paper_assets.py`
- 投稿一键打包脚本：`paper/scripts/build_submission_package.py`
- 投稿清单自动验收脚本：`paper/scripts/generate_submission_readiness.py`
- 导出的论文图表索引：`paper/assets_manifest.json`
- 扩展附录图表索引：`paper/appendix_assets/extended_assets_manifest.json`

## 方法说明（经济学）

1. 主干 DID / Event Study / DR-DID
- 主处理变量：`treated_city_direct_core`
- 主因变量：仅允许具物理含义的原始观测变量或自然对数版本，如 `log_viirs_ntl`、`physical_built_expansion_primary`、`knowledge_capital_raw`
- 综合指数仅用于 Dashboard 展示、排序和预测辅助，不作为主因果回归的 `Y`
- 控制变量白名单：`temperature_mean`、`precipitation_sum`、`baseline_population_log`
- 标准误：按 `iso3` 聚类

2. 宏观分解与识别边界
- `gdp_disaggregated_by_ntl` 仅由 `viirs_ntl_sum` 的国家内占比生成
- OSM 道路、建筑、POI 不得参与 GDP 降尺度，避免“用基建分 GDP，再用基建解释 GDP”的循环论证

3. 合成控制 / 反事实
- 反事实轨迹保留为设计敏感性与辅助验证模块
- 任何基于综合指数的旧输出，只作为历史兼容产物和辅助诊断，不作为当前主因果 headline 结论

4. 外部效度（External Validity）
- 外部结果变量：`patent_residents`、`researchers_per_million`、`high_tech_exports_share`、`employment_rate`、`urban_population_share`、`electricity_access`、`fixed_broadband_subscriptions`、`pm25_exposure`，并构建 `night_lights_proxy`（电力+宽带客观代理）
- 检验方式：国家-年份 TWFE + 时序预测增益（R2 uplift）+ 年度秩相关

## 工程规范

- 全模块使用类型注解与 docstring
- API 请求实现重试退避与缓存复用
- I/O 结果统一落盘到 `data/`、`models/`、`reports/`
- 日志分阶段输出（采集、建模、计量、导出）

## 学术诚信与引用边界

- 本项目只参考公开方法的“思想与识别框架”，不复制第三方论文文本与代码实现。
- 每个核心方法的“参考来源 -> 本地实现文件 -> 原创边界”见 `paper/source_method_attribution_ledger.md`。
- 政策事件分为 `external_direct`（外部可核验）、`objective_indicator`（WB 指标结构突变事件）、`objective_macro`（客观宏观规则代理）和 `ai_inferred`（模型推断补齐）四条轨道，并在结果中分轨报告，避免将推断事件当作事实事件陈述。
- 对外投稿建议同时提交：
  - `paper/source_method_attribution_ledger.md`
  - `data/processed/source_audit_summary.json`
  - `data/processed/policy_event_registry_audit.json`
