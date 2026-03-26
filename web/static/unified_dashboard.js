(function () {
  const state = {
    year: "all",
    continent: "all",
    tab: "overview",
  };

  const charts = {
    nowcast: null,
    gate: null,
    quadrant: null,
    evidence: null,
  };

  function num(v, fallback = NaN) {
    const x = Number(v);
    return Number.isFinite(x) ? x : fallback;
  }

  function fmt(v, d = 3) {
    const x = num(v);
    return Number.isFinite(x) ? x.toFixed(d) : "N/A";
  }

  function humanize(s) {
    return String(s || "N/A").replaceAll("_", " ");
  }

  function esc(s) {
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll("\"", "&quot;")
      .replaceAll("'", "&#39;");
  }

  function getEl(id) {
    return document.getElementById(id);
  }

  function setText(id, text) {
    const el = getEl(id);
    if (!el) return;
    el.textContent = text;
  }

  function destroyChart(name) {
    if (charts[name]) {
      charts[name].destroy();
      charts[name] = null;
    }
  }

  function getInitialTab() {
    const b = document.body;
    const fromBody = (b && b.dataset && b.dataset.initialTab) ? b.dataset.initialTab : "overview";
    const fromQuery = new URLSearchParams(window.location.search).get("tab");
    return fromQuery || fromBody || "overview";
  }

  function applyTab(tabName) {
    state.tab = tabName;
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      const active = btn.dataset.tab === tabName;
      btn.classList.toggle("active", active);
    });
    document.querySelectorAll(".tab-panel").forEach((panel) => {
      panel.classList.toggle("active", panel.id === `tab-${tabName}`);
    });
  }

  function bindTabs() {
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        applyTab(btn.dataset.tab || "overview");
      });
    });
  }

  function fillSelect(selectId, values, selectedValue, labelMap) {
    const el = getEl(selectId);
    if (!el) return;
    const prev = String(selectedValue ?? "all");
    const list = Array.isArray(values) ? values : [];
    el.replaceChildren();
    const allOption = document.createElement("option");
    allOption.value = "all";
    allOption.textContent = "全部";
    el.appendChild(allOption);
    list.forEach((v) => {
      const key = String(v);
      const label = typeof labelMap === "function" ? labelMap(v) : key;
      const option = document.createElement("option");
      option.value = key;
      option.textContent = String(label);
      el.appendChild(option);
    });
    el.value = prev;
    if (el.value !== prev) el.value = "all";
  }

  function renderKpis(bundle) {
    const k = (bundle && bundle.kpis) || {};
    const ready = Boolean(k.gate_ready);
    setText("k-gate", ready ? "READY" : "NOT READY");
    const gateEl = getEl("k-gate");
    if (gateEl) gateEl.className = ready ? "good" : "bad";
    setText("k-evidence", fmt(k.evidence_score_0_100, 2));
    setText("k-dir", fmt(k.nowcast_directional_accuracy, 3));
    setText("k-fragile", `${fmt(100 * num(k.fragile_boom_share, NaN), 1)}%`);
  }

  function renderNowcast(rows) {
    const data = (Array.isArray(rows) ? rows : [])
      .slice()
      .sort((a, b) => num(a.year, 0) - num(b.year, 0));
    const labels = data.map((r) => String(r.year));
    const obs = data.map((r) => num(r.is_forecast, 0) === 1 ? null : num(r.dynamic_pulse_index_mean, NaN));
    const fc = data.map((r) => num(r.forecast_mean, NaN));
    const lo = data.map((r) => num(r.forecast_ci_low_95, NaN));
    const hi = data.map((r) => num(r.forecast_ci_high_95, NaN));

    const ctx = getEl("nowcast-chart");
    if (!ctx) return;
    destroyChart("nowcast");
    charts.nowcast = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Observed", data: obs, borderColor: "#0f766e", tension: 0.24, spanGaps: true },
          { label: "Nowcast", data: fc, borderColor: "#1d4ed8", borderDash: [6, 4], tension: 0.24, spanGaps: true },
          { label: "CI Low", data: lo, borderColor: "#94a3b8", pointRadius: 0, borderWidth: 1, tension: 0.2 },
          { label: "CI High", data: hi, borderColor: "#94a3b8", pointRadius: 0, borderWidth: 1, tension: 0.2 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        scales: {
          y: { min: 0, max: 100, grid: { color: "#d7e3df" } },
          x: { grid: { display: false } },
        },
      },
    });
  }

  function renderGate(rows) {
    const data = (Array.isArray(rows) ? rows : []).slice();
    const labels = data.map((r) => humanize(r.gate));
    const vals = data.map((r) => Math.abs(num(r.gap_ratio, 0)));
    const pass = data.map((r) => num(r.passed, 0));
    const ctx = getEl("gate-chart");
    if (!ctx) return;
    destroyChart("gate");
    charts.gate = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          data: vals,
          label: "|Gap Ratio|",
          borderRadius: 6,
          backgroundColor: pass.map((p) => p === 1 ? "#0f766ebb" : "#b91c1ccc"),
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { beginAtZero: true, grid: { color: "#d7e3df" } },
          y: { grid: { display: false } },
        },
      },
    });
  }

  function renderQuadrant(rows) {
    const data = (Array.isArray(rows) ? rows : []).slice();
    const grouped = new Map();
    data.forEach((r) => {
      const regime = String(r.trajectory_regime || "other");
      if (!grouped.has(regime)) grouped.set(regime, []);
      grouped.get(regime).push({
        x: num(r.acceleration_score, 0),
        y: num(r.stall_risk_score, 0),
        r: Math.max(3, Math.min(12, 3 + 0.1 * num(r.dynamic_pulse_index, 0))),
      });
    });
    const palette = ["#0f766ebb", "#1d4ed8bb", "#0ea5a6bb", "#b45309bb", "#dc2626bb", "#64748bbb"];
    const datasets = Array.from(grouped.entries()).slice(0, 6).map((entry, i) => ({
      label: humanize(entry[0]),
      data: entry[1],
      backgroundColor: palette[i % palette.length],
    }));

    const ctx = getEl("quadrant-chart");
    if (!ctx) return;
    destroyChart("quadrant");
    charts.quadrant = new Chart(ctx, {
      type: "bubble",
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { min: 0, max: 100, title: { display: true, text: "Acceleration" }, grid: { color: "#d7e3df" } },
          y: { min: 0, max: 100, title: { display: true, text: "Stall Risk" }, grid: { color: "#d7e3df" } },
        },
      },
    });
  }

  function renderEvidence(rows) {
    const ranked = (Array.isArray(rows) ? rows : [])
      .slice()
      .sort((a, b) => num(b.score_0_100, 0) - num(a.score_0_100, 0))
      .slice(0, 8);
    const labels = ranked.map((r) => humanize(r.evidence_track));
    const vals = ranked.map((r) => num(r.score_0_100, 0));
    const ctx = getEl("evidence-chart");
    if (!ctx) return;
    destroyChart("evidence");
    charts.evidence = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          data: vals,
          borderRadius: 6,
          backgroundColor: vals.map((v) => v >= 80 ? "#0f766ecc" : (v >= 65 ? "#b45309cc" : "#b91c1ccc")),
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: { min: 0, max: 100, grid: { color: "#d7e3df" } },
          x: { ticks: { maxRotation: 65, minRotation: 25 }, grid: { display: false } },
        },
      },
    });
  }

  function renderRiskTable(rows) {
    const el = getEl("risk-body");
    if (!el) return;
    const top = (Array.isArray(rows) ? rows : [])
      .slice()
      .sort((a, b) => num(b.stall_risk_score, 0) - num(a.stall_risk_score, 0))
      .slice(0, 14);
    el.innerHTML = top.map((r) => `
      <tr>
        <td>${esc(r.city_name || "N/A")}</td>
        <td>${esc(r.country || "N/A")}</td>
        <td>${fmt(r.stall_risk_score, 1)}</td>
        <td>${fmt(r.acceleration_score, 1)}</td>
      </tr>
    `).join("");
  }

  function renderSummary(bundle) {
    const method = (bundle && bundle.method_core_summary) || {};
    const methodMain = method.global_vs_ridge_ar2 || {};
    const snapshot = (bundle && bundle.dynamic_snapshot) || {};
    const submit = (bundle && bundle.submission_readiness) || {};
    const rows = [
      ["Nowcast Coverage95", fmt((bundle.kpis || {}).nowcast_coverage95, 3)],
      ["Method MAE Gain", fmt(methodMain.mae_gain, 3)],
      ["Method p-value", fmt(methodMain.ternary_accuracy_p_value, 3)],
      ["Scope City Count", String(snapshot.city_count || "N/A")],
      ["High Risk Share", `${fmt(100 * num(snapshot.high_stall_risk_share, NaN), 1)}%`],
      ["Checklist", `${submit.done_items || 0}/${submit.total_items || 0}`],
    ];
    const el = getEl("summary-body");
    if (!el) return;
    el.innerHTML = rows.map((r) => `<tr><th>${esc(r[0])}</th><td>${esc(r[1])}</td></tr>`).join("");
  }

  function renderActions(bundle) {
    const gate = ((bundle || {}).top_tier_summary || {}).top_tier_gate || {};
    const weak = (((bundle || {}).highlights || {}).weakest_evidence_track || {});
    const strongest = (((bundle || {}).highlights || {}).strongest_primary_identification || {});
    const snapshot = (bundle || {}).dynamic_snapshot || {};
    const narrative = [
      `Gate=${gate.ready ? "READY" : "NOT READY"}，通过率=${fmt(gate.gate_pass_rate, 3)}。`,
      `脆弱繁荣占比=${fmt(100 * num(snapshot.fragile_boom_share, NaN), 1)}%，高失速占比=${fmt(100 * num(snapshot.high_stall_risk_share, NaN), 1)}%。`,
      `最弱证据轨道=${humanize(weak.evidence_track || "N/A")}。`,
      `最强主轨识别=${humanize(strongest.design_variant || "N/A")}，强度=${fmt(strongest.identification_strength, 2)}。`,
    ].join(" ");
    setText("narrative-text", narrative);

    const actions = [];
    const failed = (bundle.gate_checks || []).filter((r) => num(r.passed, 0) !== 1);
    failed.slice(0, 3).forEach((r) => {
      actions.push(`修复 ${humanize(r.gate)}：当前=${fmt(r.value, 4)}，阈值=${fmt(r.threshold, 4)}。`);
    });
    if (weak && Object.keys(weak).length) {
      actions.push(`优先补强 ${humanize(weak.evidence_track)} 证据轨道。`);
    }
    if (num(snapshot.fragile_boom_share, 0) >= 0.25) {
      actions.push("补做脆弱繁荣亚群异质性机制检验。");
    }
    if (!actions.length) {
      actions.push("Gate 全通过，继续补充外部效度与机制附录。");
    }
    const el = getEl("actions-list");
    if (!el) return;
    el.innerHTML = actions.map((x) => `<li>${esc(x)}</li>`).join("");
  }

  async function loadBundle() {
    const qs = new URLSearchParams();
    if (state.continent && state.continent !== "all") qs.set("continent", state.continent);
    if (state.year && state.year !== "all") qs.set("year", state.year);
    const url = qs.toString() ? `/api/frontend_bundle?${qs.toString()}` : "/api/frontend_bundle";
    const resp = await fetch(url);
    return resp.json();
  }

  async function renderAll() {
    const bundle = await loadBundle();
    const active = (bundle && bundle.active_filters) || {};
    const options = (bundle && bundle.filter_options) || {};

    state.continent = String(active.continent || state.continent || "all");
    state.year = active.year === null || active.year === undefined ? "all" : String(active.year);

    fillSelect("continent-select", options.continents || [], state.continent);
    fillSelect("year-select", options.years || [], state.year, (x) => String(x));

    renderKpis(bundle);
    renderNowcast(bundle.nowcast_global || []);
    renderGate(bundle.gate_checks || []);
    renderQuadrant(bundle.dynamic_index_rows || []);
    renderEvidence(bundle.evidence_convergence || []);
    renderRiskTable(bundle.dynamic_index_rows || []);
    renderSummary(bundle);
    renderActions(bundle);
  }

  function bindControls() {
    const year = getEl("year-select");
    const continent = getEl("continent-select");
    const refresh = getEl("refresh-btn");
    if (year) {
      year.addEventListener("change", () => {
        state.year = year.value || "all";
        renderAll();
      });
    }
    if (continent) {
      continent.addEventListener("change", () => {
        state.continent = continent.value || "all";
        renderAll();
      });
    }
    if (refresh) {
      refresh.addEventListener("click", () => renderAll());
    }
  }

  async function boot() {
    state.tab = getInitialTab();
    bindTabs();
    bindControls();
    applyTab(state.tab);
    await renderAll();
  }

  boot();
})();
