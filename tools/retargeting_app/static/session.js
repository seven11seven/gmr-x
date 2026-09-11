/**
 * Persist Stage 1–5 input selections across page navigations.
 * Keys: xml, bvh, bvh_unit, config
 *
 * Stage 1 → 2 is a full page load. Capture on select change, on stepper
 * click, and on pagehide, and also pass values via the query string so the
 * next stage can restore even if localStorage was empty.
 */
(function (global) {
  const KEY = "gmr.retarget.inputs";

  function $(id) {
    return document.getElementById(id);
  }

  function load() {
    try {
      return JSON.parse(localStorage.getItem(KEY) || "null") || {};
    } catch (_) {
      return {};
    }
  }

  function save(partial) {
    const next = Object.assign(load(), partial);
    localStorage.setItem(KEY, JSON.stringify(next));
    return next;
  }

  function fromQuery() {
    const q = new URLSearchParams(location.search);
    const out = {};
    if (q.has("xml")) out.xml = q.get("xml") || "";
    if (q.has("bvh")) out.bvh = q.get("bvh") || "";
    if (q.has("unit")) out.bvh_unit = q.get("unit") || "";
    if (q.has("bvh_unit")) out.bvh_unit = q.get("bvh_unit") || "";
    if (q.has("config")) out.config = q.get("config") || "";
    return out;
  }

  /**
   * Merge server defaults ← localStorage ← URL query (query wins).
   * Pass { optionalConfig: true } on Stage 1 so an empty config stays empty
   * unless the user (or query) actually chose one.
   */
  function pick(defaults, opts) {
    const base = {
      xml: defaults && defaults.xml,
      bvh: defaults && defaults.bvh,
      bvh_unit: defaults && defaults.bvh_unit,
    };
    if (!(opts && opts.optionalConfig)) {
      base.config = defaults && defaults.config;
    }
    return Object.assign(base, load(), fromQuery());
  }

  function setIfOption(sel, value) {
    if (!sel || value == null) return false;
    const want = String(value);
    if (![...sel.options].some((o) => o.value === want)) return false;
    sel.value = want;
    return true;
  }

  function capture() {
    const xml = $("xmlSelect");
    const bvh = $("bvhSelect");
    const unit = $("bvhUnit");
    const cfg = $("configSelect");
    const partial = {};
    if (xml) partial.xml = xml.value;
    if (bvh) partial.bvh = bvh.value;
    if (unit) partial.bvh_unit = unit.value;
    if (cfg) partial.config = cfg.value;
    if (Object.keys(partial).length) save(partial);
    return partial;
  }

  function restore() {
    const s = Object.assign(load(), fromQuery());
    if ("xml" in s) setIfOption($("xmlSelect"), s.xml);
    if ("bvh" in s) setIfOption($("bvhSelect"), s.bvh);
    if ("bvh_unit" in s) setIfOption($("bvhUnit"), s.bvh_unit);
    if ("config" in s) setIfOption($("configSelect"), s.config);
    syncSaveName();
  }

  function syncSaveName() {
    const cfg = $("configSelect");
    const saveEl = $("saveName");
    if (!cfg || !saveEl) return;
    // Stage 3 saves robot XML, not an IK config.
    if (saveEl.dataset.kind === "xml") return;
    const name = (cfg.value || "").split("/").pop();
    if (name) saveEl.value = name;
  }

  function withQuery(href) {
    const s = load();
    const url = new URL(href, location.origin);
    if (s.xml) url.searchParams.set("xml", s.xml);
    if (s.bvh) url.searchParams.set("bvh", s.bvh);
    if (s.bvh_unit != null && s.bvh_unit !== "") {
      url.searchParams.set("unit", String(s.bvh_unit));
    }
    if (s.config) url.searchParams.set("config", s.config);
    return url.pathname + url.search;
  }

  function bindStepper() {
    document.querySelectorAll(".stepper a").forEach((a) => {
      if (a._gmrSessionBound) return;
      a.addEventListener("click", (e) => {
        capture();
        const next = withQuery(a.getAttribute("href") || a.href);
        e.preventDefault();
        location.href = next;
      });
      a._gmrSessionBound = true;
    });
  }

  function bind() {
    const ids = ["xmlSelect", "bvhSelect", "bvhUnit", "configSelect"];
    for (const id of ids) {
      const el = $(id);
      if (!el || el._gmrSessionBound) continue;
      el.addEventListener("change", () => {
        capture();
        if (id === "configSelect") syncSaveName();
      });
      el._gmrSessionBound = true;
    }
    if (!window._gmrSessionPagehide) {
      window.addEventListener("pagehide", capture);
      window.addEventListener("beforeunload", capture);
      window._gmrSessionPagehide = true;
    }
    bindStepper();
    syncSaveName();
  }

  global.GmrSession = {
    load, save, capture, restore, bind, pick, syncSaveName, fromQuery,
  };
})(window);
