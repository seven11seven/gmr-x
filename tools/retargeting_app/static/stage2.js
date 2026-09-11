/* Stage-2: scale + offset tuning with root-aligned comparison */

const state = {
  meta: null,
  scene: null,
  config: null,
  configPath: null,
  robotOffset: [0, 0, 0],
  robotOrig: null, // original robot trace coords for translate
};

const AXIS_LEN = 0.06;
const C_ROBOT = "#7faac9";
const C_ROBOT_MAP = "#2ca02c";
const C_HUMAN = "#ff9896";
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

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
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
  let preferred = sel.config || d.config;
  for (const c of state.meta.ik_configs) {
    const opt = document.createElement("option");
    opt.value = c.path;
    opt.textContent = `${c.name}  (${c.dir})`;
    cfgSel.appendChild(opt);
  }
  if (preferred && [...cfgSel.options].some((o) => o.value === preferred)) {
    cfgSel.value = preferred;
  }
  if (window.GmrSession) {
    GmrSession.restore();
    GmrSession.bind();
  }

  $("btnReload").addEventListener("click", () => reloadFromDisk());
  $("btnApply").addEventListener("click", () => applyCurrent(false));
  const btnApplyOffset = $("btnApplyOffset");
  if (btnApplyOffset) {
    btnApplyOffset.addEventListener("click", () => applyCurrent(false));
  }
  $("btnSave").addEventListener("click", saveConfig);
  $("btnPreview").addEventListener("click", previewConfig);
  $("btnAutoEstimate").addEventListener("click", () => runAutoEstimate(null));
  $("btnAmbiguityCancel").addEventListener("click", () => {
    $("ambiguityDialog").close();
  });
  $("btnAmbiguityOk").addEventListener("click", () => {
    const resolutions = collectAmbiguityResolutions();
    $("ambiguityDialog").close();
    runAutoEstimate(resolutions);
  });
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

  // Save name follows the loaded IK config (GmrSession.bind).

  await reloadFromDisk();
}

async function reloadFromDisk() {
  $("btnReload").disabled = true;
  $("saveStatus").textContent = "";
  $("saveStatus").classList.remove("err");
  try {
    state.configPath = $("configSelect").value;
    state.robotOffset = [0, 0, 0];
    // Load config from disk — do not pass in-memory edits
    const data = await fetchJSON("/api/stage2/scene", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        xml: $("xmlSelect").value,
        bvh: $("bvhSelect").value,
        config_path: state.configPath,
        bvh_unit: Number($("bvhUnit").value),
      }),
    });
    ingestScene(data);
  } catch (err) {
    setError(err);
  } finally {
    $("btnReload").disabled = false;
  }
}

async function applyCurrent(fromDisk, opts = {}) {
  $("btnApply").disabled = true;
  if ($("btnApplyOffset")) $("btnApplyOffset").disabled = true;
  $("saveStatus").textContent = "";
  $("saveStatus").classList.remove("err");
  try {
    const scaleOnly = !!opts.scaleOnly;
    const body = {
      xml: $("xmlSelect").value,
      bvh: $("bvhSelect").value,
      config_path: state.configPath,
      bvh_unit: Number($("bvhUnit").value),
      config: fromDisk ? undefined : (state.config || undefined),
      scale_vertical: collectScales().vertical,
      scale_horizontal: collectScales().horizontal,
      // Scale 自动刷新时不读取 Offset DOM，避免半成品旋转矩阵被提交/冲掉
      offset_rows: scaleOnly ? undefined : collectOffsets(),
      robot_root_name: state.config?.robot_root_name,
      human_root_name: state.config?.human_root_name,
    };
    const data = await fetchJSON("/api/stage2/scene", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    ingestScene(data, {
      keepTranslate: true,
      keepOffsetPanel: scaleOnly,
      keepScalePanel: false,
    });
  } catch (err) {
    setError(err);
  } finally {
    $("btnApply").disabled = false;
    if ($("btnApplyOffset")) $("btnApplyOffset").disabled = false;
  }
}

/** Scale 改动后自动刷新；Offset 不走这个路径 */
const debouncedScaleApply = debounce(() => applyCurrent(false, { scaleOnly: true }), 450);

function setError(err) {
  $("saveStatus").textContent = String(err.message || err);
  $("saveStatus").classList.add("err");
}

function ingestScene(data, opts = {}) {
  state.scene = data.scene;
  state.config = data.config;
  if (data.config_path) state.configPath = data.config_path;
  if (data.scene.bvh_unit != null) {
    $("bvhUnit").value = String(data.scene.bvh_unit);
  }
  if (window.GmrSession) GmrSession.capture();

  const h = data.scene.detected_human_height;
  const ratio = data.scene.height_ratio;
  const unit = data.scene.bvh_unit;
  if (h == null) {
    $("heightHint").textContent = `单位 ÷${unit} · 未能检测身高 · ratio=${ratio?.toFixed(3)}`;
  } else if (h < 0.5 || h > 3.0) {
    $("heightHint").textContent =
      `身高 ${h.toFixed(3)} m（异常）· ÷${unit} — 请检查单位制`;
    $("heightHint").style.color = "#f0a040";
  } else {
    $("heightHint").textContent =
      `身高 ≈ ${h.toFixed(3)} m · assumption ${data.scene.human_height_assumption} · ratio=${ratio.toFixed(4)} · ÷${unit}`;
    $("heightHint").style.color = "";
  }

  if (!opts.keepScalePanel) {
    renderScalePanel(data.scene);
  }
  if (!opts.keepOffsetPanel) {
    renderOffsetPanel(data.scene.offset_rows || [], data.scene);
  }
  drawScene(data.scene);
  if (!opts.keepTranslate) state.robotOffset = [0, 0, 0];
  applyRobotTranslate();
}

/** Depth-first order of bones from parent→child edges. */
function dfsBoneOrder(nodeNames, edges, rootName) {
  const nameSet = new Set(nodeNames);
  const children = new Map();
  const hasParent = new Set();
  for (const e of edges || []) {
    const p = e[0], c = e[1];
    if (!children.has(p)) children.set(p, []);
    children.get(p).push(c);
    hasParent.add(c);
  }
  for (const [, kids] of children) kids.sort();

  const order = [];
  const visited = new Set();
  function dfs(name) {
    if (visited.has(name)) return;
    visited.add(name);
    order.push(name);
    for (const ch of (children.get(name) || [])) dfs(ch);
  }

  const roots = [];
  if (rootName && nameSet.has(rootName)) roots.push(rootName);
  for (const n of nodeNames) {
    if (!hasParent.has(n) && n !== rootName) roots.push(n);
  }
  // deterministic leftover roots
  roots.sort((a, b) => {
    if (a === rootName) return -1;
    if (b === rootName) return 1;
    return a.localeCompare(b);
  });
  // de-dupe roots while preserving order
  const seenRoot = new Set();
  for (const r of roots) {
    if (seenRoot.has(r)) continue;
    seenRoot.add(r);
    dfs(r);
  }
  for (const n of nodeNames) dfs(n);
  return order.filter((n) => nameSet.has(n));
}

function boneOrderIndexMap(scene, which) {
  const skel = which === "robot" ? scene.robot : scene.human;
  const names = (skel?.nodes || []).map((n) => n.name);
  const root = which === "robot"
    ? scene.robot_root_name
    : scene.human_root_name;
  const order = dfsBoneOrder(names, skel?.edges || [], root);
  const idx = new Map();
  order.forEach((n, i) => idx.set(n, i));
  return idx;
}

function renderScalePanel(scene) {
  const v = scene.scale_vertical || {};
  const h = scene.scale_horizontal || {};
  const allNames = [...new Set([
    ...Object.keys(v),
    ...Object.keys(h),
    ...(scene.human?.nodes || []).map((n) => n.name),
  ].filter(Boolean))];

  const dfs = dfsBoneOrder(
    (scene.human?.nodes || []).map((n) => n.name),
    scene.human?.edges || [],
    scene.human_root_name,
  );
  const dfsIdx = new Map(dfs.map((n, i) => [n, i]));
  // DFS first; bones missing from tree go last (alpha)
  const names = allNames.slice().sort((a, b) => {
    const ia = dfsIdx.has(a) ? dfsIdx.get(a) : 1e9;
    const ib = dfsIdx.has(b) ? dfsIdx.get(b) : 1e9;
    if (ia !== ib) return ia - ib;
    return a.localeCompare(b);
  });

  const list = $("scaleList");
  list.innerHTML = "";

  const header = document.createElement("div");
  header.className = "scale-row";
  header.innerHTML = `<span class="name">bone</span><span>V</span><span>H</span>`;
  header.style.opacity = "0.6";
  list.appendChild(header);

  for (const name of names) {
    const row = document.createElement("div");
    row.className = "scale-row";
    row.dataset.bone = name;
    const nm = document.createElement("span");
    nm.className = "name";
    nm.textContent = name;
    nm.title = name;
    const iv = document.createElement("input");
    iv.type = "number";
    iv.step = "0.01";
    iv.value = Number(v[name] ?? 1.0).toFixed(3);
    iv.dataset.kind = "v";
    const ih = document.createElement("input");
    ih.type = "number";
    ih.step = "0.01";
    ih.value = Number(h[name] ?? v[name] ?? 1.0).toFixed(3);
    ih.dataset.kind = "h";
    iv.addEventListener("change", debouncedScaleApply);
    ih.addEventListener("change", debouncedScaleApply);
    row.append(nm, iv, ih);
    list.appendChild(row);
  }
}

function collectScales() {
  const vertical = {};
  const horizontal = {};
  for (const row of document.querySelectorAll("#scaleList .scale-row[data-bone]")) {
    const name = row.dataset.bone;
    const inputs = row.querySelectorAll("input");
    vertical[name] = Number(inputs[0].value);
    horizontal[name] = Number(inputs[1].value);
  }
  return { vertical, horizontal };
}

function renderOffsetPanel(rows, scene) {
  const list = $("offsetList");
  list.innerHTML = "";
  if (!rows.length) {
    list.innerHTML = `<p class="hint">当前 config 无 ik_match_table1 条目</p>`;
    return;
  }

  const humanIdx = boneOrderIndexMap(scene || state.scene || {}, "human");
  const robotIdx = boneOrderIndexMap(scene || state.scene || {}, "robot");
  const sorted = rows.slice().sort((a, b) => {
    const ha = humanIdx.has(a.human_bone) ? humanIdx.get(a.human_bone) : 1e9;
    const hb = humanIdx.has(b.human_bone) ? humanIdx.get(b.human_bone) : 1e9;
    if (ha !== hb) return ha - hb;
    const ra = robotIdx.has(a.robot_body) ? robotIdx.get(a.robot_body) : 1e9;
    const rb = robotIdx.has(b.robot_body) ? robotIdx.get(b.robot_body) : 1e9;
    if (ra !== rb) return ra - rb;
    return String(a.robot_body).localeCompare(String(b.robot_body));
  });

  for (const row of sorted) {
    list.appendChild(buildOffsetCard(row));
  }
}

function buildOffsetCard(row) {
  const card = document.createElement("div");
  card.className = "offset-card";
  card.dataset.robot = row.robot_body;
  card.dataset.human = row.human_bone;
  card.dataset.posWeight = row.pos_weight;
  card.dataset.rotWeight = row.rot_weight;

  const title = document.createElement("h3");
  title.innerHTML = `${row.robot_body} <span class="sub">→ ${row.human_bone}</span>`;
  card.appendChild(title);

  const xyz = document.createElement("div");
  xyz.className = "xyz-row";
  ["x", "y", "z"].forEach((axis, i) => {
    const lab = document.createElement("label");
    lab.textContent = `pos ${axis}`;
    const inp = document.createElement("input");
    inp.type = "number";
    inp.step = "0.001";
    inp.className = `pos-${axis}`;
    inp.value = Number(row.pos_offset[i] || 0).toFixed(4);
    // Offset：不自动刷新，改完后点「应用当前参数」
    lab.appendChild(inp);
    xyz.appendChild(lab);
  });
  card.appendChild(xyz);

  const matLabel = document.createElement("div");
  matLabel.className = "mat-label";
  matLabel.textContent = "旋转矩阵 R（改完后点「应用当前参数」）";
  card.appendChild(matLabel);

  const mat = document.createElement("div");
  mat.className = "mat-grid";
  const M = row.rot_matrix || [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const inp = document.createElement("input");
      inp.type = "number";
      inp.step = "0.001";
      inp.dataset.r = r;
      inp.dataset.c = c;
      inp.className = "mat-cell";
      inp.value = Number(M[r][c]).toFixed(4);
      // 仅本地更新四元数提示，不触发场景刷新
      inp.addEventListener("change", () => { refreshQuatHint(card); });
      mat.appendChild(inp);
    }
  }
  card.appendChild(mat);

  const hint = document.createElement("p");
  hint.className = "quat-hint";
  hint.textContent = `quat wxyz = [${(row.rot_quat_wxyz || []).map((x) => Number(x).toFixed(4)).join(", ")}]`;
  card.appendChild(hint);

  const idBtn = document.createElement("button");
  idBtn.type = "button";
  idBtn.className = "ghost";
  idBtn.style.cssText = "margin-top:0.35rem;padding:0.25rem 0.5rem;font-size:0.7rem;";
  idBtn.textContent = "重置为单位矩阵";
  idBtn.addEventListener("click", () => {
    const cells = card.querySelectorAll(".mat-cell");
    const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
    cells.forEach((el, i) => { el.value = I[i].toFixed(4); });
    refreshQuatHint(card);
  });
  card.appendChild(idBtn);

  return card;
}

async function refreshQuatHint(card) {
  const matrix = readMatrix(card);
  try {
    const data = await fetchJSON("/api/stage2/rot_convert", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ matrix }),
    });
    const q = data.quat_wxyz;
    card.querySelector(".quat-hint").textContent =
      `quat wxyz ≈ [${q.map((x) => Number(x).toFixed(4)).join(", ")}]（预览，点应用后生效）`;
    // 不回写正交化矩阵，避免编辑过程中单元格被覆盖
  } catch (_) {
    const hint = card.querySelector(".quat-hint");
    if (hint) hint.textContent = "quat wxyz = (矩阵未完成或非法，改完后点应用)";
  }
}

function readMatrix(card) {
  const m = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  card.querySelectorAll(".mat-cell").forEach((el) => {
    m[Number(el.dataset.r)][Number(el.dataset.c)] = Number(el.value);
  });
  return m;
}

function collectOffsets() {
  const rows = [];
  for (const card of document.querySelectorAll("#offsetList .offset-card")) {
    rows.push({
      robot_body: card.dataset.robot,
      human_bone: card.dataset.human,
      pos_weight: Number(card.dataset.posWeight || 10),
      rot_weight: Number(card.dataset.rotWeight || 10),
      pos_offset: [
        Number(card.querySelector(".pos-x").value),
        Number(card.querySelector(".pos-y").value),
        Number(card.querySelector(".pos-z").value),
      ],
      rot_matrix: readMatrix(card),
    });
  }
  return rows;
}

function edgeSegments(nodesByName, edges, offset) {
  const xs = [], ys = [], zs = [];
  for (const [a, b] of edges) {
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
    const m = quatToMat(n.quat_wxyz);
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

function drawScene(scene) {
  const robotMapped = new Set(scene.robot_mapped || []);
  const humanMapped = new Set(scene.human_mapped || []);
  const robotNodes = scene.robot.nodes;
  const visHuman = filterHumanSkeletonForVis(scene.human.nodes, scene.human.edges);
  const humanNodes = visHuman.nodes;
  // Roots already aligned at origin — no side-by-side offset
  const offset = [0, 0, 0];

  const robotByName = Object.fromEntries(robotNodes.map((n) => [n.name, n]));
  const humanByName = Object.fromEntries(humanNodes.map((n) => [n.name, n]));

  const traces = [];
  const robotTraceStart = 0;

  traces.push({
    type: "scatter3d", mode: "markers+text", name: "robot links",
    legendgroup: "robot",
    x: robotNodes.map((n) => n.pos[0]),
    y: robotNodes.map((n) => n.pos[1]),
    z: robotNodes.map((n) => n.pos[2]),
    text: robotNodes.map((n) => n.name),
    textposition: "top center",
    textfont: { size: 9, color: "#dce6f0" },
    marker: {
      size: 5,
      color: robotNodes.map((n) => (robotMapped.has(n.name) ? C_ROBOT_MAP : C_ROBOT)),
      line: { width: 1, color: "#111" },
    },
    hovertemplate: "<b>%{text}</b><br>robot<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
  });

  const rEdges = edgeSegments(robotByName, scene.robot.edges, [0, 0, 0]);
  traces.push({
    type: "scatter3d", mode: "lines", name: "robot bones", legendgroup: "robot",
    x: rEdges.x, y: rEdges.y, z: rEdges.z,
    line: { color: C_BONE, width: 3 }, hoverinfo: "skip",
  });

  const rAxes = axisSegments(robotNodes, [0, 0, 0]);
  for (const [key, color, label] of [
    ["x", "#e74c3c", "robot X"], ["y", "#27ae60", "robot Y"], ["z", "#3498db", "robot Z"],
  ]) {
    traces.push({
      type: "scatter3d", mode: "lines", name: label,
      legendgroup: "robot_axes", showlegend: key === "x",
      x: rAxes[key].x, y: rAxes[key].y, z: rAxes[key].z,
      line: { color, width: 3 }, hoverinfo: "skip",
    });
  }

  const robotTraceEnd = traces.length;

  traces.push({
    type: "scatter3d", mode: "markers+text", name: "BVH bones",
    legendgroup: "human",
    x: humanNodes.map((n) => n.pos[0] + offset[0]),
    y: humanNodes.map((n) => n.pos[1] + offset[1]),
    z: humanNodes.map((n) => n.pos[2] + offset[2]),
    text: humanNodes.map((n) => n.name),
    textposition: "top center",
    textfont: { size: 9, color: "#f5e6d8" },
    marker: {
      size: 5,
      color: humanNodes.map((n) => (humanMapped.has(n.name) ? C_HUMAN_MAP : C_HUMAN)),
      line: { width: 1, color: "#111" },
    },
    hovertemplate: "<b>%{text}</b><br>human<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
  });

  const hEdges = edgeSegments(humanByName, visHuman.edges, offset);
  traces.push({
    type: "scatter3d", mode: "lines", name: "human bones", legendgroup: "human",
    x: hEdges.x, y: hEdges.y, z: hEdges.z,
    line: { color: "#c07040", width: 3 }, hoverinfo: "skip",
  });

  const hAxes = axisSegments(humanNodes, offset);
  for (const [key, color, label] of [
    ["x", "#e74c3c", "human X"], ["y", "#27ae60", "human Y"], ["z", "#3498db", "human Z"],
  ]) {
    traces.push({
      type: "scatter3d", mode: "lines", name: label,
      legendgroup: "human_axes", showlegend: key === "x",
      x: hAxes[key].x, y: hAxes[key].y, z: hAxes[key].z,
      line: { color, width: 3 }, hoverinfo: "skip",
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
    margin: { l: 0, r: 0, t: 40, b: 0 },
    title: {
      text: `${scene.xml_path.split("/").pop()}  ↔  ${scene.bvh_path.split("/").pop()}  |  roots @ origin`,
      font: { size: 13 },
    },
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
    uirevision: "gmr-stage2",
  };

  Plotly.react("plot", traces, layout, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["toImage", "lasso2d", "select2d"],
  });

  // Cache original robot trace coords for translation
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

function savePayload() {
  const scales = collectScales();
  return {
    config_path: state.configPath,
    base_config: state.config,
    filename: $("saveName").value.trim() || "stage2_config.json",
    overwrite: $("overwriteSrc").checked,
    save_to_ik_configs: $("saveToIk").checked,
    bvh_unit: Number($("bvhUnit").value),
    scale_vertical: scales.vertical,
    scale_horizontal: scales.horizontal,
    offset_rows: collectOffsets(),
    robot_root_name: state.config?.robot_root_name,
    human_root_name: state.config?.human_root_name,
  };
}

async function previewConfig() {
  try {
    // Build via save endpoint logic without writing: reuse stage2 scene merge
    const body = savePayload();
    const data = await fetchJSON("/api/stage2/scene", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        xml: $("xmlSelect").value,
        bvh: $("bvhSelect").value,
        config_path: body.config_path,
        config: body.base_config,
        scale_vertical: body.scale_vertical,
        scale_horizontal: body.scale_horizontal,
        offset_rows: body.offset_rows,
        bvh_unit: body.bvh_unit,
      }),
    });
    $("jsonPreview").textContent = JSON.stringify(data.config, null, 4);
    $("jsonDialog").showModal();
  } catch (err) {
    setError(err);
  }
}

async function saveConfig() {
  $("saveStatus").classList.remove("err");
  try {
    const data = await fetchJSON("/api/stage2/config/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(savePayload()),
    });
    $("saveStatus").textContent = `已保存: ${data.path}`;
    state.config = data.config;
    state.configPath = data.path;
    // refresh config list
    state.meta = await fetchJSON("/api/meta");
    const cur = data.path;
    const cfgSel = $("configSelect");
    cfgSel.innerHTML = "";
    for (const c of state.meta.ik_configs) {
      const opt = document.createElement("option");
      opt.value = c.path;
      opt.textContent = `${c.name}  (${c.dir})`;
      cfgSel.appendChild(opt);
    }
    if ([...cfgSel.options].some((o) => o.value === cur)) cfgSel.value = cur;
    if (window.GmrSession) {
      GmrSession.capture();
      GmrSession.syncSaveName();
    }
  } catch (err) {
    setError(err);
  }
}

function estimatePayload(resolutions) {
  const scales = collectScales();
  return {
    xml: $("xmlSelect").value,
    bvh: $("bvhSelect").value,
    config_path: state.configPath,
    config: state.config,
    bvh_unit: Number($("bvhUnit").value),
    scale_vertical: scales.vertical,
    scale_horizontal: scales.horizontal,
    offset_rows: collectOffsets(),
    robot_root_name: state.config?.robot_root_name,
    human_root_name: state.config?.human_root_name,
    resolutions: resolutions || {},
  };
}

async function runAutoEstimate(resolutions) {
  const hint = $("estimateHint");
  hint.textContent = "估计中…";
  hint.style.color = "";
  $("btnAutoEstimate").disabled = true;
  try {
    const data = await fetchJSON("/api/stage2/auto_estimate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(estimatePayload(resolutions)),
    });

    if (data.needs_resolution) {
      showAmbiguityDialog(data.ambiguities || data.all_ambiguities || []);
      hint.textContent = "请先在弹窗中选择一对多映射关系";
      hint.style.color = "#f0a040";
      return;
    }

    applyEstimateToPanels(data);
    hint.textContent = formatEstimateMeta(data.meta);
    // 刷新可视化（含 offset）
    await applyCurrent(false);
  } catch (err) {
    hint.textContent = String(err.message || err);
    hint.style.color = "#f07178";
    setError(err);
  } finally {
    $("btnAutoEstimate").disabled = false;
  }
}

function formatEstimateMeta(meta) {
  if (!meta?.scale?.ratios) return "估计完成";
  const r = meta.scale.ratios;
  const raw = meta.scale.ratios_raw;
  const hr = meta.scale.height_ratio;
  let msg = `估计完成 · V下=${r.lower_vertical.toFixed(3)} V上=${r.upper_vertical.toFixed(3)} · H下=${r.lower_horizontal.toFixed(3)} H上=${r.upper_horizontal.toFixed(3)}`;
  if (hr != null) {
    msg += ` · 已补偿 height_ratio=${Number(hr).toFixed(3)}`;
  }
  if (raw) {
    msg += ` (raw V下=${raw.lower_vertical.toFixed(3)})`;
  }
  return msg;
}

function applyEstimateToPanels(data) {
  // Update scale inputs
  const v = data.scale_vertical || {};
  const h = data.scale_horizontal || {};
  for (const row of document.querySelectorAll("#scaleList .scale-row[data-bone]")) {
    const name = row.dataset.bone;
    const inputs = row.querySelectorAll("input");
    if (v[name] != null) inputs[0].value = Number(v[name]).toFixed(3);
    if (h[name] != null) inputs[1].value = Number(h[name]).toFixed(3);
  }
  // Re-render offset panel with new rot matrices (pos kept from server rows)
  if (data.offset_rows && state.scene) {
    // merge: keep current pos from DOM if present
    const posByRobot = {};
    for (const card of document.querySelectorAll("#offsetList .offset-card")) {
      posByRobot[card.dataset.robot] = [
        Number(card.querySelector(".pos-x").value),
        Number(card.querySelector(".pos-y").value),
        Number(card.querySelector(".pos-z").value),
      ];
    }
    const rows = data.offset_rows.map((r) => ({
      ...r,
      pos_offset: posByRobot[r.robot_body] || r.pos_offset,
    }));
    renderOffsetPanel(rows, state.scene);
  }

  // Sync into state.config scales for subsequent apply
  if (state.config) {
    state.config.human_scale_table_vertical = { ...v };
    state.config.human_scale_table_horizontal = { ...h };
  }
}

function showAmbiguityDialog(ambiguities) {
  const list = $("ambiguityList");
  list.innerHTML = "";
  if (!ambiguities.length) {
    list.innerHTML = "<p class='hint'>无歧义项</p>";
  }
  for (const a of ambiguities) {
    const box = document.createElement("div");
    box.className = "amb-item";
    const title = document.createElement("div");
    title.className = "amb-title";
    const sel = document.createElement("select");
    sel.className = "amb-select";

    if (a.kind === "robot_to_many_humans") {
      title.textContent = `机器人 ${a.robot_body} 对应多个人类骨骼 → 选择人类骨骼`;
      sel.dataset.resKey = a.robot_body;
      sel.dataset.kind = a.kind;
      for (const c of a.candidates) {
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        sel.appendChild(opt);
      }
    } else if (a.kind === "human_to_many_robots") {
      title.textContent = `人类 ${a.human_bone} 被多个机器人关节映射 → 选择用于估计旋转的机器人关节`;
      sel.dataset.resKey = `human::${a.human_bone}`;
      sel.dataset.kind = a.kind;
      for (const c of a.candidates) {
        const opt = document.createElement("option");
        opt.value = c;
        opt.textContent = c;
        sel.appendChild(opt);
      }
    } else {
      continue;
    }
    box.append(title, sel);
    list.appendChild(box);
  }
  $("ambiguityDialog").showModal();
}

function collectAmbiguityResolutions() {
  const resolutions = {};
  for (const sel of document.querySelectorAll("#ambiguityList .amb-select")) {
    resolutions[sel.dataset.resKey] = sel.value;
  }
  return resolutions;
}

init().catch((err) => setError(err));
