(function () {
  const KEY = "urban_pulse_dashboard_filters_v1";
  const DEFAULT_STATE = { year: "all", continent: "all" };
  const FALLBACK_CONTINENTS = [
    "all",
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "South America",
    "Oceania",
  ];

  let state = { ...DEFAULT_STATE };
  const listeners = [];

  function normalizeYear(v) {
    if (v === null || v === undefined) return "all";
    const s = String(v).trim().toLowerCase();
    if (!s || s === "all") return "all";
    const n = Number(s);
    return Number.isFinite(n) ? String(Math.trunc(n)) : "all";
  }

  function normalizeContinent(v) {
    if (v === null || v === undefined) return "all";
    const s = String(v).trim();
    if (!s) return "all";
    return s.toLowerCase() === "all" ? "all" : s;
  }

  function readStorage() {
    try {
      const raw = localStorage.getItem(KEY);
      if (!raw) return { ...DEFAULT_STATE };
      const obj = JSON.parse(raw);
      return {
        year: normalizeYear(obj && obj.year),
        continent: normalizeContinent(obj && obj.continent),
      };
    } catch (_e) {
      return { ...DEFAULT_STATE };
    }
  }

  function readUrl() {
    const u = new URL(window.location.href);
    return {
      year: normalizeYear(u.searchParams.get("year")),
      continent: normalizeContinent(u.searchParams.get("continent")),
    };
  }

  function writeStorage() {
    try {
      localStorage.setItem(KEY, JSON.stringify(state));
    } catch (_e) {
      // ignore
    }
  }

  function writeUrl() {
    const u = new URL(window.location.href);
    u.searchParams.set("year", state.year);
    u.searchParams.set("continent", state.continent);
    window.history.replaceState({}, "", u.toString());
  }

  function syncNavLinks() {
    const links = document.querySelectorAll(".nav a[href]");
    links.forEach((a) => {
      const href = a.getAttribute("href");
      if (!href || href.startsWith("http") || href.startsWith("#")) return;
      try {
        const u = new URL(href, window.location.origin);
        u.searchParams.set("year", state.year);
        u.searchParams.set("continent", state.continent);
        a.setAttribute("href", `${u.pathname}${u.search}`);
      } catch (_e) {
        // ignore malformed href
      }
    });
  }

  function setState(nextState, notify = true) {
    state = {
      year: normalizeYear(nextState && nextState.year),
      continent: normalizeContinent(nextState && nextState.continent),
    };
    writeStorage();
    writeUrl();
    syncNavLinks();
    if (notify) listeners.forEach((fn) => fn({ ...state }));
  }

  function onChange(fn) {
    if (typeof fn === "function") listeners.push(fn);
  }

  function getState() {
    return { ...state };
  }

  async function discoverYears() {
    try {
      const rows = await fetch("/api/global").then((r) => r.json());
      const ys = Array.isArray(rows)
        ? rows.map((r) => Number(r.year)).filter((y) => Number.isFinite(y))
        : [];
      const uniq = Array.from(new Set(ys)).sort((a, b) => a - b);
      if (uniq.length > 0) return uniq.map((y) => String(y));
    } catch (_e) {
      // ignore
    }
    try {
      const d = await fetch("/api/dashboard_summary").then((r) => r.json());
      if (d && Number.isFinite(Number(d.latest_year))) {
        return [String(Math.trunc(Number(d.latest_year)))];
      }
    } catch (_e) {
      // ignore
    }
    return [];
  }

  async function discoverContinents() {
    try {
      const rows = await fetch("/api/realtime/continents").then((r) => r.json());
      const cs = Array.isArray(rows)
        ? rows.map((r) => String(r.continent || "").trim()).filter((x) => !!x)
        : [];
      const uniq = Array.from(new Set(cs));
      if (uniq.length > 0) return ["all", ...uniq];
    } catch (_e) {
      // ignore
    }
    return [...FALLBACK_CONTINENTS];
  }

  function buildFilterStrip({ years, continents, hint }) {
    const strip = document.createElement("div");
    strip.className = "filter-strip";
    strip.id = "up-filter-strip";
    strip.innerHTML = `
      <div class="filter-group">
        <label for="up-filter-year">Year</label>
        <select id="up-filter-year"></select>
      </div>
      <div class="filter-group">
        <label for="up-filter-continent">Continent</label>
        <select id="up-filter-continent"></select>
      </div>
      <div class="filter-actions">
        <button id="up-filter-apply" type="button">应用筛选</button>
        <button id="up-filter-reset" type="button" class="ghost">重置</button>
      </div>
      <p class="filter-note">${hint || "筛选状态在各子页面共享。部分页面可能只使用其中一个维度。"}</p>
    `;

    const ySel = strip.querySelector("#up-filter-year");
    const cSel = strip.querySelector("#up-filter-continent");
    const applyBtn = strip.querySelector("#up-filter-apply");
    const resetBtn = strip.querySelector("#up-filter-reset");

    const yearOptions = ["all", ...(years || [])];
    ySel.innerHTML = yearOptions
      .map((y) => `<option value="${y}">${y === "all" ? "All Years" : y}</option>`)
      .join("");
    cSel.innerHTML = (continents || FALLBACK_CONTINENTS)
      .map((c) => `<option value="${c}">${c === "all" ? "All Continents" : c}</option>`)
      .join("");

    ySel.value = yearOptions.includes(state.year) ? state.year : "all";
    cSel.value = (continents || FALLBACK_CONTINENTS).includes(state.continent) ? state.continent : "all";

    applyBtn.addEventListener("click", () => {
      setState({ year: ySel.value, continent: cSel.value }, true);
    });
    resetBtn.addEventListener("click", () => {
      ySel.value = "all";
      cSel.value = "all";
      setState({ year: "all", continent: "all" }, true);
    });
    ySel.addEventListener("change", () => setState({ year: ySel.value, continent: cSel.value }, true));
    cSel.addEventListener("change", () => setState({ year: ySel.value, continent: cSel.value }, true));
    return strip;
  }

  async function mount(options = {}) {
    const fromStorage = readStorage();
    const fromUrl = readUrl();
    const initial = {
      year: fromUrl.year !== "all" ? fromUrl.year : fromStorage.year,
      continent: fromUrl.continent !== "all" ? fromUrl.continent : fromStorage.continent,
    };
    state = {
      year: normalizeYear(initial.year),
      continent: normalizeContinent(initial.continent),
    };

    const container = document.querySelector(options.containerSelector || ".hero");
    const years = await discoverYears();
    const continents = await discoverContinents();
    if (container && !document.getElementById("up-filter-strip")) {
      const strip = buildFilterStrip({
        years,
        continents,
        hint: options.hint,
      });
      container.appendChild(strip);
    }
    if (typeof options.onChange === "function") onChange(options.onChange);
    setState(state, false);
    return getState();
  }

  window.UPDashboardFilters = {
    mount,
    getState,
    setState,
    onChange,
  };
})();
