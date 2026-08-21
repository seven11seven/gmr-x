/* Stage-4: full-trajectory retarget + MuJoCo frame sync */

const state = {
  meta: null,
  config: null,
  configPath: null,
  frames: [],
  robotEdges: [],
  humanEdges: [],
  frameIdx: 0,
  mode: "preview",
  robotOffset: [0, 0, 0],
  robotOrig: null,
  robotTraceIndices: null,
  fps: 30,
  playing: false,
  playTimer: null,
  pollTimer: null,
  jobId: null,
};

const AXIS_LEN = 0.05;
const C_ROBOT_MAP = "#2ca02c";
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

  fillSelect($("xmlSelect"), state.meta.xml_files, "path", (x) => x.path, d.xml);
  fillSelect($("bvhSelect"), state.meta.bvh_files, "path", (x) => x.path, d.bvh);
  fillSelect(
    $("bvhUnit"),
    state.meta.bvh_units,
    "unit",
    (u) => `${u.label}  (scale=${u.scale})`,
    String(d.bvh_unit),
  );

  const cfgSel = $("configSelect");
  cfgSel.innerHTML = "";
  for (const c of state.meta.ik_configs) {
    const opt = document.createElement("option");
    opt.value = c.path;
    opt.textContent = `${c.name}  (${c.dir})`;
    cfgSel.appendChild(opt);
  }
  if (d.config && [...cfgSel.options].some((o) => o.value === d.config)) {
    cfgSel.value = d.config;
  }
  if (cfgSel.value) $("saveName").value = cfgSel.value.split("/").pop();

  cfgSel.addEventListener("change", () => {
    $("saveName").value = cfgSel.value.split("/").pop() || "stage4_config.json";
  });

  $("btnReload").addEventListener("click", reloadWeights);
  $("btnSave").addEventListener("click", saveConfig);
  $("btnRetarget").addEventListener("click", runRetarget);
  $("btnPlay").addEventListener("click", () => setPlaying(true));
  $("btnPause").addEventListener("click", () => setPlaying(false));
  $("btnTransReset").addEventListener("click", () => {
    state.robotOffset = [0, 0, 0];
    applyRobotTranslate();
  });
  $("frameSlider").addEventListener("input", () => {
    goToFrame(Number($("frameSlider").value), true);
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

  await reloadWeights();
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
    if (name) flashWeightRow(name);
  });
  plotDiv._gmrClickBound = true;
}

async function reloadWeights() {
  $("btnReload").disabled = true;
  $("saveStatus").textContent = "";
  $("saveStatus").classList.remove("err");
  try {
    state.configPath = $("configSelect").value;
    const data = await fetchJSON("/api/stage3/weights", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config_path: state.configPath }),
    });
    state.config = data.config;
    if (data.config?.bvh_unit != null) {
      $("bvhUnit").value = String(data.config.bvh_unit);
    }
    state._pendingWeightRows = data.weight_rows || [];
    $("retargetHint").textContent = "Config 已加载。正在加载 scale/offset 预览…";
    await loadPreviewScene();
    const sorted = sortWeightsByRobotDfs(
      state._pendingWeightRows,
      state.frames[0]?.robot,
      state.config?.robot_root_name,
    );
    renderWeightPanel(sorted);
    $("retargetHint").textContent =
      "已显示 scale/offset 对比。点击「开始重定向并显示」处理全轨迹。";
  } catch (err) {
    setError(err);
  } finally {
    $("btnReload").disabled = false;
  }
}

async function loadPreviewScene() {
  const data = await fetchJSON("/api/stage2/scene", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      xml: $("xmlSelect").value,
      bvh: $("bvhSelect").value,
      config_path: state.configPath,
      config: state.config,
      bvh_unit: Number($("bvhUnit").value),
    }),
  });
  const scene = data.scene;
  state.config = data.config || state.config;
  state.mode = "preview";
  state.robotEdges = scene.robot?.edges || [];
  state.humanEdges = scene.human?.edges || [];
  state.frames = [{
    robot: { nodes: scene.robot.nodes },
    human: { nodes: scene.human.nodes },
  }];
  state.frameIdx = 0;
  state.robotOffset = [0, 0, 0];
  stopLocalPlay();
  $("btnPlay").disabled = true;
  $("btnPause").disabled = true;
  $("frameTotal").textContent = "0";
  $("frameIndex").textContent = "0";
  const slider = $("frameSlider");
  slider.min = "0";
  slider.max = "0";
  slider.value = "0";
  slider.disabled = true;
  drawCurrentFrame();
}

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
  let root = rootName && nameSet.has(rootName) ? rootName : null;
  if (!root) {
    for (const n of nodeNames) {
      if (!hasParent.has(n)) { root = n; break; }
    }
  }
  if (!root && nodeNames.length) root = nodeNames[0];
  const order = [];
  const seen = new Set();
  function dfs(n) {
    if (seen.has(n)) return;
    seen.add(n);
    order.push(n);
    for (const c of children.get(n) || []) dfs(c);
  }
  if (root) dfs(root);
  for (const n of nodeNames) dfs(n);
  return order;
}

function sortWeightsByRobotDfs(rows, robotSkel, rootName) {
  if (!rows?.length) return [];
  if (!robotSkel?.nodes?.length) {
    return [...rows].sort((a, b) =>
      String(a.robot_body).localeCompare(String(b.robot_body)));
  }
  const names = robotSkel.nodes.map((n) => n.name);
  let edges = robotSkel.edges || state.robotEdges || [];
  if ((!edges || !edges.length) && robotSkel.nodes.some((n) => n.parent)) {
    edges = robotSkel.nodes
      .filter((n) => n.parent)
      .map((n) => [n.parent, n.name]);
  }
  const order = dfsBoneOrder(names, edges, rootName);
  const idx = new Map(order.map((n, i) => [n, i]));
  return [...rows].sort((a, b) => {
    const ia = idx.has(a.robot_body) ? idx.get(a.robot_body) : 1e9;
    const ib = idx.has(b.robot_body) ? idx.get(b.robot_body) : 1e9;
    if (ia !== ib) return ia - ib;
    return String(a.robot_body).localeCompare(String(b.robot_body));
  });
}

function flashWeightRow(robotName) {
  const el = document.querySelector(
    `#weightList .weight-row[data-robot="${CSS.escape(robotName)}"]`,
  );
  if (!el) return;
  el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  el.classList.add("flash");
  setTimeout(() => el.classList.remove("flash"), 900);
}

function renderWeightPanel(rows) {
  const list = $("weightList");
  list.innerHTML = "";
  const header = document.createElement("div");
  header.className = "weight-row weight-head";
  header.innerHTML = `<span class="name">robot → human</span><span>pos</span><span>rot</span>`;
  list.appendChild(header);
  for (const row of rows) {
    const el = document.createElement("div");
    el.className = "weight-row";
    el.dataset.robot = row.robot_body;
    el.dataset.human = row.human_bone;
    el.id = `weight-${row.robot_body}`;
    const nm = document.createElement("span");
    nm.className = "name";
    nm.title = `${row.robot_body} → ${row.human_bone}`;
    nm.innerHTML = `${row.robot_body}<br><span class="sub">→ ${row.human_bone}</span>`;
    const ip = document.createElement("input");
    ip.className = "pos-w";
    ip.type = "number";
    ip.step = "0.1";
    ip.value = String(row.pos_weight);
    const ir = document.createElement("input");
    ir.className = "rot-w";
    ir.type = "number";
    ir.step = "0.1";
    ir.value = String(row.rot_weight);
    el.append(nm, ip, ir);
    list.appendChild(el);
  }
}

function collectWeights() {
  const rows = [];
  for (const el of document.querySelectorAll("#weightList .weight-row[data-robot]")) {
    rows.push({
      robot_body: el.dataset.robot,
      human_bone: el.dataset.human,
      pos_weight: Number(el.querySelector(".pos-w").value),
      rot_weight: Number(el.querySelector(".rot-w").value),
    });
  }
  return rows;
}

function setProgress(cur, total) {
  const wrap = $("progressWrap");
  const bar = $("progressBar");
  if (!total) {
    wrap.hidden = true;
    return;
  }
  wrap.hidden = false;
  const pct = Math.min(100, Math.round((100 * cur) / total));
  bar.style.width = `${pct}%`;
}

async function runRetarget() {
  $("btnRetarget").disabled = true;
  $("btnPlay").disabled = true;
  $("btnPause").disabled = true;
  stopLocalPlay();
  $("retargetHint").textContent = "正在启动全轨迹重定向…";
  $("retargetHint").style.color = "";
  $("saveStatus").classList.remove("err");
  setProgress(0, 1);

  try {
    const start = await fetchJSON("/api/stage4/retarget", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        xml: $("xmlSelect").value,
        bvh: $("bvhSelect").value,
        config_path: state.configPath,
        config: state.config,
        bvh_unit: Number($("bvhUnit").value),
        weight_rows: collectWeights(),
      }),
    });
    state.jobId = start.job_id;
    await pollJobUntilDone();
  } catch (err) {
    $("retargetHint").textContent = String(err.message || err);
    $("retargetHint").style.color = "#f07178";
    setError(err);
    setProgress(0, 0);
    $("btnRetarget").disabled = false;
  }
}

function pollJobUntilDone() {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const st = await fetchJSON("/api/stage4/status");
        if (st.total) {
          setProgress(st.progress || 0, st.total);
          $("retargetHint").textContent =
            `重定向中 ${st.progress || 0}/${st.total} · ${st.message || ""}`;
        }
        if (st.status === "running") {
          state.pollTimer = setTimeout(tick, 400);
          return;
        }
        if (st.status === "error") {
          throw new Error(st.error || "retarget failed");
        }
        if (st.status === "done") {
          await loadRetargetResult(st);
          resolve();
          return;
        }
        // idle unexpected
        state.pollTimer = setTimeout(tick, 400);
      } catch (err) {
        reject(err);
      }
    };
    tick();
  }).finally(() => {
    $("btnRetarget").disabled = false;
  });
}

async function loadRetargetResult(statusMeta) {
  const data = await fetchJSON("/api/stage4/frames");
  state.config = data.config || state.config;
  state.robotEdges = data.robot_edges || [];
  state.humanEdges = data.human_edges || [];
  state.frames = (data.frames || []).map((f) => ({
    robot: { nodes: f.robot.nodes, edges: state.robotEdges },
    human: { nodes: f.human.nodes, edges: state.humanEdges },
  }));
  state.fps = data.fps || statusMeta.fps || 30;
  state.frameIdx = 0;
  state.mode = "retarget";
  state.robotOffset = [0, 0, 0];

  const n = state.frames.length;
  $("frameTotal").textContent = String(Math.max(n - 1, 0));
  $("frameIndex").textContent = "0";
  const slider = $("frameSlider");
  slider.min = "0";
  slider.max = String(Math.max(n - 1, 0));
  slider.value = "0";
  slider.disabled = n <= 1;
  $("btnPlay").disabled = n <= 1;
  $("btnPause").disabled = n <= 1;

  // Attach edges onto first frame robot for weight DFS
  if (state.frames[0]) {
    state.frames[0].robot.edges = state.robotEdges;
  }
  const sorted = sortWeightsByRobotDfs(
    data.weight_rows || collectWeights(),
    state.frames[0]?.robot,
    state.config?.robot_root_name,
  );
  renderWeightPanel(sorted);

  drawCurrentFrame();
  await syncFrameToServer(0, false);

  let hint =
    `完成：${n} 帧 @ ${Number(state.fps).toFixed(1)} fps · Web 与 MuJoCo 已同步`;
  if (data.viewer_error || statusMeta.viewer_error) {
    hint += ` · MuJoCo 启动失败: ${data.viewer_error || statusMeta.viewer_error}`;
  } else if (data.viewer_alive === false) {
    hint += " · MuJoCo viewer 未就绪（检查本机 DISPLAY）";
  } else {
    hint += " · 已打开 MuJoCo 窗口";
  }
  $("retargetHint").textContent = hint;
  setProgress(n, n);
  setTimeout(() => setProgress(0, 0), 800);
}

async function syncFrameToServer(idx, playing) {
  try {
    await fetchJSON("/api/stage4/sync", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame_idx: idx, playing }),
    });
  } catch (_) {
    /* viewer may be closed; ignore */
  }
}

function goToFrame(idx, syncServer) {
  if (!state.frames.length) return;
  idx = Math.max(0, Math.min(idx, state.frames.length - 1));
  state.frameIdx = idx;
  $("frameSlider").value = String(idx);
  $("frameIndex").textContent = String(idx);
  drawCurrentFrame();
  if (syncServer && state.mode === "retarget") {
    syncFrameToServer(idx, state.playing);
  }
}

function stopLocalPlay() {
  state.playing = false;
  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
  }
}

async function setPlaying(on) {
  if (state.mode !== "retarget" || state.frames.length <= 1) return;
  state.playing = !!on;
  await syncFrameToServer(state.frameIdx, state.playing);

  if (state.playTimer) {
    clearInterval(state.playTimer);
    state.playTimer = null;
  }
  if (!state.playing) return;

  const dt = Math.max(8, Math.round(1000 / Math.max(1, state.fps)));
  state.playTimer = setInterval(() => {
    if (!state.playing) return;
    let nxt = state.frameIdx + 1;
    if (nxt >= state.frames.length) {
      stopLocalPlay();
      syncFrameToServer(state.frameIdx, false);
      return;
    }
    // Local advance; MuJoCo also auto-advances when playing=true on server.
    // Keep web as source of truth by syncing each frame.
    goToFrame(nxt, true);
  }, dt);
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

function drawCurrentFrame() {
  if (!state.frames.length) {
    Plotly.purge("plot");
    return;
  }
  const frame = state.frames[state.frameIdx];
  const robotNodes = frame.robot.nodes;
  const humanNodes = frame.human.nodes;
  const robotByName = Object.fromEntries(robotNodes.map((n) => [n.name, n]));
  const humanByName = Object.fromEntries(humanNodes.map((n) => [n.name, n]));
  const rEdgeList = frame.robot.edges || state.robotEdges || [];
  const hEdgeList = frame.human.edges || state.humanEdges || [];

  const traces = [];
  const robotTraceStart = 0;

  traces.push({
    type: "scatter3d", mode: "markers+text", name: "robot",
    legendgroup: "robot",
    x: robotNodes.map((n) => n.pos[0]),
    y: robotNodes.map((n) => n.pos[1]),
    z: robotNodes.map((n) => n.pos[2]),
    text: robotNodes.map((n) => n.name),
    textposition: "top center",
    textfont: { size: 8, color: "#dce6f0" },
    marker: { size: 4, color: C_ROBOT_MAP, line: { width: 1, color: "#111" } },
    hovertemplate: "<b>%{text}</b><br>robot<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
  });

  const rEdges = edgeSegments(robotByName, rEdgeList, [0, 0, 0]);
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
      line: { color, width: 2 }, hoverinfo: "skip",
    });
  }
  const robotTraceEnd = traces.length;

  traces.push({
    type: "scatter3d", mode: "markers+text", name: "human target",
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

  const hEdges = edgeSegments(humanByName, hEdgeList, [0, 0, 0]);
  traces.push({
    type: "scatter3d", mode: "lines", name: "human bones", legendgroup: "human",
    x: hEdges.x, y: hEdges.y, z: hEdges.z,
    line: { color: "#c07040", width: 3 }, hoverinfo: "skip",
  });

  const hAxes = axisSegments(humanNodes, [0, 0, 0]);
  for (const [key, color, label] of [
    ["x", "#e74c3c", "human X"], ["y", "#27ae60", "human Y"], ["z", "#3498db", "human Z"],
  ]) {
    traces.push({
      type: "scatter3d", mode: "lines", name: label,
      legendgroup: "human_axes", showlegend: key === "x",
      x: hAxes[key].x, y: hAxes[key].y, z: hAxes[key].z,
      line: { color, width: 2, dash: "dot" }, hoverinfo: "skip",
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
    title: {
      text: state.mode === "retarget"
        ? `Full Retarget · Frame ${state.frameIdx} / ${Math.max(state.frames.length - 1, 0)}`
        : "Preview · scale + offset（未重定向）",
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
    uirevision: "gmr-stage4",
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

async function saveConfig() {
  $("saveStatus").classList.remove("err");
  try {
    const data = await fetchJSON("/api/stage3/config/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        config_path: state.configPath,
        base_config: state.config,
        filename: $("saveName").value.trim() || "stage4_config.json",
        overwrite: $("overwriteSrc").checked,
        save_to_ik_configs: $("saveToIk").checked,
        weight_rows: collectWeights(),
      }),
    });
    $("saveStatus").textContent = `已保存: ${data.path}`;
    state.config = data.config;
    state.configPath = data.path;

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
  } catch (err) {
    setError(err);
  }
}

init().catch((err) => setError(err));
