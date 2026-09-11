/* Stage-3: fine-tune aux_* local positions in robot XML */

const state = {
  meta: null,
  config: null,
  configPath: null,
  xmlPath: null,
  auxRows: [],
  robot: null,
  human: null,
  robotOffset: [0, 0, 0],
  robotOrig: null,
  robotTraceIndices: null,
  refreshTimer: null,
};

const AXIS_LEN = 0.05;
const C_ROBOT = "#7faac9";
const C_ROBOT_MAP = "#2ca02c";
const C_AUX = "#f0c040";
const C_HUMAN_MAP = "#ff7f0e";
const C_BONE = "#888888";

function $(id) { return document.getElementById(id); }

function fillSelect(sel, items, valueKey, labelFn, selected) {
  sel.innerHTML = "";
  for (const item of items) {
    const opt = document.createElement("option");
    opt.value = typeof item === "string" ? item : item[valueKey];
    opt.textContent = labelFn ? labelFn(item) : opt.value;
    sel.appendChild(opt);
  }
  if (selected) sel.value = selected;
}

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(data.error || res.statusText);
  }
  return data;
}

function setError(err) {
  $("saveStatus").textContent = String(err.message || err);
  $("saveStatus").classList.add("err");
}

async function init() {
  state.meta = await fetchJSON("/api/meta");
  const d = state.meta.defaults;
  const sel = window.GmrSession ? GmrSession.pick(d) : d;

  fillSelect($("xmlSelect"), state.meta.xml_files, "path", (x) => x.path, sel.xml || d.xml);
  fillSelect($("bvhSelect"), state.meta.bvh_files, "path", (x) => x.path, sel.bvh || d.bvh);
  fillSelect(
    $("bvhUnit"),
    state.meta.bvh_units,
    "unit",
    (u) => `${u.label}  (scale=${u.scale})`,
    String(sel.bvh_unit ?? d.bvh_unit),
  );
  [...$("bvhUnit").options].forEach((o) => { o.value = String(o.value); });

  const cfgSel = $("configSelect");
  cfgSel.innerHTML = "";
  const empty = document.createElement("option");
  empty.value = "";
  empty.textContent = "（不加载 config / 仅机器人）";
  cfgSel.appendChild(empty);
  for (const c of state.meta.ik_configs) {
    const opt = document.createElement("option");
    opt.value = c.path;
    opt.textContent = `${c.name}  (${c.dir})`;
    cfgSel.appendChild(opt);
  }
  const preferred = sel.config || d.config;
  if (preferred && [...cfgSel.options].some((o) => o.value === preferred)) {
    cfgSel.value = preferred;
  }
  if (window.GmrSession) {
    GmrSession.restore();
    GmrSession.bind();
  }

  $("btnReload").addEventListener("click", () => reloadScene(true));
  $("btnSave").addEventListener("click", saveXml);
  $("btnTransReset").addEventListener("click", () => {
    state.robotOffset = [0, 0, 0];
    applyRobotTranslate();
  });
  document.querySelectorAll(".translate-grid button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const axis = btn.dataset.axis;
      const dir = Number(btn.dataset.dir);
      const step = Number($("transStep").value) || 0.05;
      const idx = { x: 0, y: 1, z: 2 }[axis];
      state.robotOffset[idx] += dir * step;
      applyRobotTranslate();
    });
  });

  await reloadScene(true);
}

function collectAuxPositions() {
  const out = {};
  for (const el of document.querySelectorAll("#auxList .aux-row[data-name]")) {
    const name = el.dataset.name;
    const xs = el.querySelectorAll("input.aux-xyz");
    out[name] = [
      Number(xs[0].value) || 0,
      Number(xs[1].value) || 0,
      Number(xs[2].value) || 0,
    ];
  }
  return out;
}

function scheduleRefresh() {
  if (state.refreshTimer) clearTimeout(state.refreshTimer);
  state.refreshTimer = setTimeout(() => reloadScene(false), 180);
}

function flashAuxRow(name) {
  const el = document.querySelector(
    `#auxList .aux-row[data-name="${CSS.escape(name)}"]`,
  );
  if (!el) return;
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  window.setTimeout(() => el.classList.remove("flash"), 900);
}

function renderAuxPanel(rows) {
  const list = $("auxList");
  list.innerHTML = "";
  if (!rows?.length) {
    list.innerHTML = `<p class="hint">当前 XML 中没有 aux_* 辅助点</p>`;
    return;
  }

  const header = document.createElement("div");
  header.className = "weight-row weight-header aux-row-head";
  header.innerHTML = `<span class="name">aux link</span><span>x</span><span>y</span><span>z</span>`;
  list.appendChild(header);

  for (const row of rows) {
    const el = document.createElement("div");
    el.className = "weight-row aux-row";
    el.dataset.name = row.name;
    el.id = `aux-${row.name}`;

    const nm = document.createElement("span");
    nm.className = "name";
    nm.title = row.parent ? `parent: ${row.parent}` : row.name;
    nm.innerHTML = `${row.name}${
      row.parent ? `<br><span class="sub">⊂ ${row.parent}</span>` : ""
    }`;

    const inputs = [0, 1, 2].map((i) => {
      const ip = document.createElement("input");
      ip.type = "number";
      ip.step = "0.001";
      ip.className = "aux-xyz";
      ip.value = String(row.pos[i]);
      ip.addEventListener("input", scheduleRefresh);
      ip.addEventListener("change", scheduleRefresh);
      return ip;
    });

    el.append(nm, ...inputs);
    list.appendChild(el);
  }
}

async function reloadScene(resetPanel) {
  $("btnReload").disabled = true;
  $("loadHint").textContent = "加载中…";
  $("loadHint").style.color = "";
  try {
    state.xmlPath = $("xmlSelect").value;
    state.configPath = $("configSelect").value || null;

    let config = null;
    if (state.configPath) {
      const w = await fetchJSON("/api/stage4/weights", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config_path: state.configPath }),
      });
      config = w.config;
      state.config = config;
      if (config?.bvh_unit != null) $("bvhUnit").value = String(config.bvh_unit);
    } else {
      state.config = null;
    }

    const aux_positions = resetPanel ? null : collectAuxPositions();
    const data = await fetchJSON("/api/stage3/aux_scene", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        xml: state.xmlPath,
        bvh: $("bvhSelect").value,
        config_path: state.configPath,
        config,
        bvh_unit: Number($("bvhUnit").value),
        aux_positions,
      }),
    });

    state.robot = data.robot;
    state.human = data.human;
    state.auxRows = data.aux_rows || [];
    state.xmlPath = data.xml_path || state.xmlPath;

    if (resetPanel || !document.querySelector("#auxList .aux-row")) {
      renderAuxPanel(state.auxRows);
    }

    if (resetPanel) state.robotOffset = [0, 0, 0];
    drawScene();
    const nAux = state.auxRows.length;
    $("loadHint").textContent =
      `已加载 ${nAux} 个 aux_* · ${state.human ? "含人类对比" : "仅机器人"}`;
    if (window.GmrSession) GmrSession.capture();
  } catch (err) {
    $("loadHint").textContent = String(err.message || err);
    $("loadHint").style.color = "#f07178";
    setError(err);
  } finally {
    $("btnReload").disabled = false;
  }
}

function ensurePlotClickHandler() {
  const plotDiv = $("plot");
  if (!plotDiv || plotDiv._gmrClickBound) return;
  if (typeof plotDiv.on !== "function") return;
  plotDiv.on("plotly_click", (ev) => {
    if (!ev?.points?.length) return;
    const pt = ev.points[0];
    if (pt.curveNumber !== 0) return;
    const name = pt.text
      || (Array.isArray(pt.data?.text) ? pt.data.text[pt.pointIndex] : null);
    if (name && String(name).startsWith("aux_")) flashAuxRow(name);
  });
  plotDiv._gmrClickBound = true;
}

function edgeSegments(nodesByName, edges, offset) {
  const xs = [], ys = [], zs = [];
  for (const [a, b] of edges || []) {
    const pa = nodesByName[a];
    const pb = nodesByName[b];
    if (!pa || !pb) continue;
    xs.push(pa.pos[0] + offset[0], pb.pos[0] + offset[0], null);
    ys.push(pa.pos[1] + offset[1], pb.pos[1] + offset[1], null);
    zs.push(pa.pos[2] + offset[2], pb.pos[2] + offset[2], null);
  }
  return { x: xs, y: ys, z: zs };
}

function quatToMat(q) {
  const [w, x, y, z] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

function axisSegments(nodes, offset) {
  const segs = { x: { x: [], y: [], z: [] }, y: { x: [], y: [], z: [] }, z: { x: [], y: [], z: [] } };
  const keys = ["x", "y", "z"];
  for (const n of nodes) {
    const m = quatToMat(n.quat_wxyz || [1, 0, 0, 0]);
    const p = [n.pos[0] + offset[0], n.pos[1] + offset[1], n.pos[2] + offset[2]];
    for (let i = 0; i < 3; i++) {
      const end = [p[0] + m[0][i] * AXIS_LEN, p[1] + m[1][i] * AXIS_LEN, p[2] + m[2][i] * AXIS_LEN];
      const s = segs[keys[i]];
      s.x.push(p[0], end[0], null);
      s.y.push(p[1], end[1], null);
      s.z.push(p[2], end[2], null);
    }
  }
  return segs;
}

function drawScene() {
  if (!state.robot?.nodes?.length) {
    Plotly.purge("plot");
    return;
  }
  const robotNodes = state.robot.nodes;
  const visHuman = filterHumanSkeletonForVis(
    state.human?.nodes || [],
    state.human?.edges || [],
  );
  const humanNodes = visHuman.nodes;
  const robotByName = Object.fromEntries(robotNodes.map((n) => [n.name, n]));
  const humanByName = Object.fromEntries(humanNodes.map((n) => [n.name, n]));

  const traces = [];
  const robotTraceStart = 0;
  const colors = robotNodes.map((n) => (n.is_aux ? C_AUX : C_ROBOT_MAP));

  traces.push({
    type: "scatter3d", mode: "markers+text", name: "robot",
    legendgroup: "robot",
    x: robotNodes.map((n) => n.pos[0]),
    y: robotNodes.map((n) => n.pos[1]),
    z: robotNodes.map((n) => n.pos[2]),
    text: robotNodes.map((n) => n.name),
    textposition: "top center",
    textfont: { size: 8, color: "#dce6f0" },
    marker: { size: 4, color: colors, line: { width: 1, color: "#111" } },
    hovertemplate: "<b>%{text}</b><br>robot<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
  });

  const rEdges = edgeSegments(robotByName, state.robot.edges, [0, 0, 0]);
  traces.push({
    type: "scatter3d", mode: "lines", name: "robot bones", legendgroup: "robot",
    x: rEdges.x, y: rEdges.y, z: rEdges.z,
    line: { color: C_BONE, width: 3 }, hoverinfo: "skip",
  });

  const auxNodes = robotNodes.filter((n) => n.is_aux);
  const rAxes = axisSegments(auxNodes.length ? auxNodes : robotNodes, [0, 0, 0]);
  for (const [key, color, label] of [
    ["x", "#e74c3c", "aux X"], ["y", "#27ae60", "aux Y"], ["z", "#3498db", "aux Z"],
  ]) {
    traces.push({
      type: "scatter3d", mode: "lines", name: label,
      legendgroup: "robot_axes", showlegend: key === "x",
      x: rAxes[key].x, y: rAxes[key].y, z: rAxes[key].z,
      line: { color, width: 2 }, hoverinfo: "skip",
    });
  }
  const robotTraceEnd = traces.length;

  if (humanNodes.length) {
    traces.push({
      type: "scatter3d", mode: "markers+text", name: "human",
      legendgroup: "human",
      x: humanNodes.map((n) => n.pos[0]),
      y: humanNodes.map((n) => n.pos[1]),
      z: humanNodes.map((n) => n.pos[2]),
      text: humanNodes.map((n) => n.name),
      textposition: "top center",
      textfont: { size: 8, color: "#f5e6d8" },
      marker: { size: 4, color: C_HUMAN_MAP, line: { width: 1, color: "#111" } },
      hovertemplate: "<b>%{text}</b><br>human<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
    });
    const hEdges = edgeSegments(humanByName, visHuman.edges, [0, 0, 0]);
    traces.push({
      type: "scatter3d", mode: "lines", name: "human bones", legendgroup: "human",
      x: hEdges.x, y: hEdges.y, z: hEdges.z,
      line: { color: "#c07040", width: 3 }, hoverinfo: "skip",
    });
  }

  const all = [];
  for (const n of robotNodes) all.push(n.pos);
  for (const n of humanNodes) all.push(n.pos);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const p of all) {
    for (let i = 0; i < 3; i++) {
      min[i] = Math.min(min[i], p[i]);
      max[i] = Math.max(max[i], p[i]);
    }
  }
  const center = min.map((v, i) => (v + max[i]) / 2);
  const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) * 0.7 || 1;

  const layout = {
    paper_bgcolor: "#0b0f14",
    plot_bgcolor: "#0b0f14",
    font: { color: "#c5d0dc", family: "IBM Plex Sans, sans-serif" },
    margin: { l: 0, r: 0, t: 36, b: 0 },
    title: { text: "Aux link fine-tune · 黄点 = aux_*", font: { size: 13 } },
    scene: {
      xaxis: { title: "X", range: [center[0] - extent, center[0] + extent], backgroundcolor: "#0b0f14", gridcolor: "#243040" },
      yaxis: { title: "Y", range: [center[1] - extent, center[1] + extent], backgroundcolor: "#0b0f14", gridcolor: "#243040" },
      zaxis: { title: "Z", range: [center[2] - extent, center[2] + extent], backgroundcolor: "#0b0f14", gridcolor: "#243040" },
      aspectmode: "cube",
      camera: { eye: { x: 1.25, y: 1.25, z: 0.85 } },
    },
    legend: {
      bgcolor: "rgba(20,26,34,0.85)", bordercolor: "#2c3642",
      font: { size: 11 }, x: 0.01, y: 0.99,
    },
    uirevision: "gmr-stage3-aux",
  };

  Plotly.react("plot", traces, layout, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["toImage", "lasso2d", "select2d"],
  });
  ensurePlotClickHandler();

  state.robotOrig = [];
  state.robotTraceIndices = [];
  for (let i = robotTraceStart; i < robotTraceEnd; i++) {
    state.robotTraceIndices.push(i);
    const t = traces[i];
    state.robotOrig.push({
      x: (t.x || []).map((v) => v),
      y: (t.y || []).map((v) => v),
      z: (t.z || []).map((v) => v),
    });
  }
  applyRobotTranslate();
}

function applyRobotTranslate() {
  const o = state.robotOffset;
  $("transPos").textContent =
    `(${o[0].toFixed(2)}, ${o[1].toFixed(2)}, ${o[2].toFixed(2)})`;
  if (!state.robotOrig || !state.robotTraceIndices) return;
  const isNum = (v) => typeof v === "number" && isFinite(v);
  const newX = [], newY = [], newZ = [];
  for (const orig of state.robotOrig) {
    newX.push(orig.x.map((v) => (isNum(v) ? v + o[0] : v)));
    newY.push(orig.y.map((v) => (isNum(v) ? v + o[1] : v)));
    newZ.push(orig.z.map((v) => (isNum(v) ? v + o[2] : v)));
  }
  Plotly.restyle("plot", { x: newX, y: newY, z: newZ }, state.robotTraceIndices);
}

async function saveXml() {
  $("saveStatus").classList.remove("err");
  try {
    const overwrite = $("overwriteXml").checked;
    let dest = null;
    if (!overwrite) {
      const name = $("saveName").value.trim();
      if (!name) throw new Error("请填写另存文件名，或勾选覆盖当前 XML");
      const src = state.xmlPath || $("xmlSelect").value;
      const dir = src.includes("/") ? src.split("/").slice(0, -1).join("/") : "assets";
      dest = `${dir}/${name.endsWith(".xml") ? name : name + ".xml"}`;
    }
    const data = await fetchJSON("/api/stage3/aux_save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        xml: state.xmlPath || $("xmlSelect").value,
        dest_path: dest,
        aux_positions: collectAuxPositions(),
      }),
    });
    $("saveStatus").textContent = `已保存: ${data.path}`;
    if (data.path) {
      state.xmlPath = data.path;
      // refresh xml list if new file
      state.meta = await fetchJSON("/api/meta");
      fillSelect($("xmlSelect"), state.meta.xml_files, "path", (x) => x.path, data.path);
    }
  } catch (err) {
    setError(err);
  }
}

init().catch((err) => setError(err));
