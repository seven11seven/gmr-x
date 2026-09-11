/* Stage-1 retargeting config UI */

const state = {
  meta: null,
  scene: null,
  mapping: {}, // robotBody -> humanBone
  baseConfigPath: null,
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

async function init() {
  state.meta = await fetchJSON("/api/meta");
  const d = state.meta.defaults;
  const sel = window.GmrSession
    ? GmrSession.pick(d, { optionalConfig: true })
    : d;

  fillSelect(
    $("xmlSelect"),
    state.meta.xml_files,
    "path",
    (x) => x.path,
    sel.xml || d.xml,
  );
  fillSelect(
    $("bvhSelect"),
    state.meta.bvh_files,
    "path",
    (x) => x.path,
    sel.bvh || d.bvh,
  );
  fillSelect(
    $("bvhUnit"),
    state.meta.bvh_units,
    "unit",
    (u) => `${u.label}  (scale=${u.scale})`,
    String(sel.bvh_unit ?? d.bvh_unit),
  );
  // unit values are numbers — ensure option values stringify correctly
  [...$("bvhUnit").options].forEach((o) => { o.value = String(o.value); });

  const cfgSel = $("configSelect");
  cfgSel.innerHTML = '<option value="">— 无 —</option>';
  for (const c of state.meta.ik_configs) {
    const opt = document.createElement("option");
    opt.value = c.path;
    opt.textContent = `${c.name}  (${c.dir})`;
    cfgSel.appendChild(opt);
  }
  if (sel.config) cfgSel.value = sel.config;
  if (window.GmrSession) {
    GmrSession.restore();
    GmrSession.bind();
  }

  $("btnLoad").addEventListener("click", loadScene);
  $("btnAddMap").addEventListener("click", () => addMappingRow("", ""));
  $("btnMirrorLR").addEventListener("click", applyLeftToRightSymmetry);
  $("btnSave").addEventListener("click", saveConfig);
  $("btnPreview").addEventListener("click", previewConfig);
  $("bvhUnit").addEventListener("change", () => {
    if (state.scene) loadScene();
  });

  // Auto-load defaults
  await loadScene();
}

function robotNames() {
  return (state.scene?.robot?.nodes || []).map((n) => n.name);
}
function humanNames() {
  return (state.scene?.human?.nodes || []).map((n) => n.name);
}

function syncRootSelects() {
  const rNames = robotNames();
  const hNames = humanNames();
  const prevR = $("robotRoot").value;
  const prevH = $("humanRoot").value;
  fillSelect($("robotRoot"), rNames, null, null, prevR || guessRobotRoot(rNames));
  fillSelect($("humanRoot"), hNames, null, null, prevH || guessHumanRoot(hNames));
}

function guessRobotRoot(names) {
  for (const c of ["aux_base", "pelvis", "base_link", "root"]) {
    if (names.includes(c)) return c;
  }
  return names[0] || "";
}
function guessHumanRoot(names) {
  for (const c of ["Hips", "pelvis", "Root", "Hip"]) {
    if (names.includes(c)) return c;
  }
  return names[0] || "";
}

function mappingFromRows() {
  const map = {};
  for (const row of document.querySelectorAll("#mappingList .map-row")) {
    const robot = row.querySelector(".robot-sel").value;
    const human = row.querySelector(".human-sel").value;
    if (robot && human) map[robot] = human;
  }
  return map;
}

function renderMappingRows(mapping) {
  const list = $("mappingList");
  list.innerHTML = "";
  const entries = Object.entries(mapping || {});
  if (entries.length === 0) {
    addMappingRow("", "");
    return;
  }
  for (const [r, h] of entries) addMappingRow(r, h);
}

function addMappingRow(robot, human) {
  const list = $("mappingList");
  const row = document.createElement("div");
  row.className = "map-row";

  const pair = document.createElement("div");
  pair.className = "map-pair";

  const rSel = document.createElement("select");
  rSel.className = "robot-sel";
  fillSelect(rSel, [""].concat(robotNames()), null, (v) => v || "— robot —", robot);

  const arrow = document.createElement("span");
  arrow.className = "arrow";
  arrow.textContent = "↓";

  const hSel = document.createElement("select");
  hSel.className = "human-sel";
  fillSelect(hSel, [""].concat(humanNames()), null, (v) => v || "— bone —", human);

  const rm = document.createElement("button");
  rm.type = "button";
  rm.className = "rm";
  rm.title = "删除";
  rm.textContent = "×";
  rm.addEventListener("click", () => {
    row.remove();
    onMappingChanged();
  });

  rSel.addEventListener("change", onMappingChanged);
  hSel.addEventListener("change", onMappingChanged);

  pair.append(rSel, arrow, hSel);
  row.append(pair, rm);
  list.appendChild(row);
}

function onMappingChanged() {
  state.mapping = mappingFromRows();
  if (state.scene) drawScene(state.scene, state.mapping);
}

/** True if name has a left side token: left_*, *_left_*, LeftXxx, LEFT_*. */
function hasLeftToken(name) {
  return /(?:^|_)left(?=_|$)|Left|LEFT/.test(name || "");
}

/** left/Left/LEFT → right/Right/RIGHT (and the reverse). */
function swapSide(name, toRight) {
  const a = toRight ? "Left" : "Right";
  const b = toRight ? "Right" : "Left";
  const aa = toRight ? "LEFT" : "RIGHT";
  const bb = toRight ? "RIGHT" : "LEFT";
  const aaa = toRight ? "left" : "right";
  const bbb = toRight ? "right" : "left";
  return String(name).split(a).join(b).split(aa).join(bb).split(aaa).join(bbb);
}

/**
 * Mirror every left_* / *_left_* robot mapping onto the corresponding right
 * link (and Left* human bone → Right*). Existing right rows are overwritten.
 */
function applyLeftToRightSymmetry() {
  const status = $("mapStatus");
  if (status) {
    status.classList.remove("err");
    status.textContent = "";
  }
  const rSet = new Set(robotNames());
  const hSet = new Set(humanNames());
  const map = mappingFromRows();
  const extras = [];
  const updates = {};
  const missing = [];

  for (const [robot, human] of Object.entries(map)) {
    if (!hasLeftToken(robot)) continue;
    const rR = swapSide(robot, true);
    if (!rR || rR === robot) continue;
    if (!rSet.has(rR)) {
      missing.push(rR);
      continue;
    }
    const hR = hasLeftToken(human) ? swapSide(human, true) : human;
    if (!hR || !hSet.has(hR)) {
      missing.push(`${rR} ← ${hR || "?"}`);
      continue;
    }
    if (map[rR] == null) extras.push([rR, hR]);
    else if (map[rR] !== hR) updates[rR] = hR;
  }

  if (!extras.length && !Object.keys(updates).length) {
    if (status) {
      status.classList.add("err");
      status.textContent = missing.length
        ? `没有可写入的右侧映射。缺失: ${missing.slice(0, 4).join(", ")}`
        : "没有可对称的 left_* / *_left_* 映射";
    }
    return;
  }

  for (const row of document.querySelectorAll("#mappingList .map-row")) {
    const r = row.querySelector(".robot-sel").value;
    if (updates[r]) row.querySelector(".human-sel").value = updates[r];
  }
  for (const [r, h] of extras) addMappingRow(r, h);
  onMappingChanged();

  if (status) {
    const bits = [];
    if (extras.length) bits.push(`新增 ${extras.length}`);
    if (Object.keys(updates).length) bits.push(`更新 ${Object.keys(updates).length}`);
    status.textContent = `已对称 left → right（${bits.join("，")}）`;
  }
}

async function loadScene() {
  $("btnLoad").disabled = true;
  $("saveStatus").textContent = "";
  $("saveStatus").classList.remove("err");
  try {
    const configPath = $("configSelect").value || null;

    // If loading an existing config that stores bvh_unit, sync the selector first.
    let mappingOverride = null;
    if (configPath) {
      try {
        const cfgData = await fetchJSON(`/api/config?path=${encodeURIComponent(configPath)}`);
        if (cfgData.bvh_unit != null) {
          $("bvhUnit").value = String(cfgData.bvh_unit);
        }
        mappingOverride = cfgData.mapping || {};
        if (cfgData.config?.robot_root_name) {
          // roots applied after scene load / syncRootSelects
          state._pendingRoots = {
            robot: cfgData.config.robot_root_name,
            human: cfgData.config.human_root_name,
          };
        }
      } catch (_) {
        // fall through — scene load may still succeed without mapping
      }
    }

    const body = {
      xml: $("xmlSelect").value,
      bvh: $("bvhSelect").value,
      bvh_unit: Number($("bvhUnit").value),
      config_path: configPath,
      mapping: mappingOverride || undefined,
    };
    const data = await fetchJSON("/api/scene", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.scene = data.scene;
    state.baseConfigPath = configPath;

    if (mappingOverride) {
      state.mapping = mappingOverride;
    } else if (data.scene.loaded_config?.mapping) {
      state.mapping = data.scene.loaded_config.mapping;
    } else if (data.scene.mapping && Object.keys(data.scene.mapping).length) {
      state.mapping = data.scene.mapping;
    } else {
      state.mapping = mappingFromRows();
    }

    syncRootSelects();
    if (state._pendingRoots) {
      if (state._pendingRoots.robot) $("robotRoot").value = state._pendingRoots.robot;
      if (state._pendingRoots.human) $("humanRoot").value = state._pendingRoots.human;
      state._pendingRoots = null;
    } else {
      if (data.scene.loaded_config?.robot_root_name) {
        $("robotRoot").value = data.scene.loaded_config.robot_root_name;
      }
      if (data.scene.loaded_config?.human_root_name) {
        $("humanRoot").value = data.scene.loaded_config.human_root_name;
      }
    }

    renderMappingRows(state.mapping);
    state.mapping = mappingFromRows();

    const h = data.scene.detected_human_height;
    const unit = data.scene.bvh_unit;
    const scale = data.scene.bvh_scale;
    if (h == null) {
      $("heightHint").textContent =
        `单位 ÷${unit} (scale=${scale}) · 未能自动检测身高，请确认单位制是否正确`;
    } else if (h < 0.5 || h > 3.0) {
      $("heightHint").textContent =
        `检测到身高 ${h.toFixed(3)} m（异常）· 当前 ÷${unit} — 请切换单位制`;
      $("heightHint").style.color = "#f0a040";
    } else {
      $("heightHint").textContent =
        `检测到身高 ≈ ${h.toFixed(3)} m · 单位 ÷${unit} (scale=${scale})`;
      $("heightHint").style.color = "";
    }

    drawScene(state.scene, state.mapping);
    if (window.GmrSession) GmrSession.capture();
  } catch (err) {
    $("saveStatus").textContent = String(err.message || err);
    $("saveStatus").classList.add("err");
  } finally {
    $("btnLoad").disabled = false;
  }
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
  // wxyz → 3x3 (columns = axes)
  const [w, x, y, z] = q;
  return [
    [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
    [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
    [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
  ];
}

function axisSegments(nodes, offset) {
  const segs = {
    x: { x: [], y: [], z: [] },
    y: { x: [], y: [], z: [] },
    z: { x: [], y: [], z: [] },
  };
  const keys = ["x", "y", "z"];
  for (const n of nodes) {
    const m = quatToMat(n.quat_wxyz);
    const p = [n.pos[0] + offset[0], n.pos[1] + offset[1], n.pos[2] + offset[2]];
    for (let i = 0; i < 3; i++) {
      const end = [
        p[0] + m[0][i] * AXIS_LEN,
        p[1] + m[1][i] * AXIS_LEN,
        p[2] + m[2][i] * AXIS_LEN,
      ];
      const s = segs[keys[i]];
      s.x.push(p[0], end[0], null);
      s.y.push(p[1], end[1], null);
      s.z.push(p[2], end[2], null);
    }
  }
  return segs;
}

function computeHumanOffset(robotNodes, humanNodes) {
  const ry = robotNodes.map((n) => n.pos[1]);
  const hy = humanNodes.map((n) => n.pos[1]);
  const rw = Math.max(...ry) - Math.min(...ry);
  const hw = Math.max(...hy) - Math.min(...hy);
  const sep = Math.max(rw, hw) * 0.7 + 0.5;
  return [0, sep, 0];
}

function drawScene(scene, mapping) {
  const robotMapped = new Set(Object.keys(mapping || {}));
  const humanMapped = new Set(Object.values(mapping || {}));
  const robotNodes = scene.robot.nodes;
  const visHuman = filterHumanSkeletonForVis(scene.human.nodes, scene.human.edges);
  const humanNodes = visHuman.nodes;
  const offset = computeHumanOffset(robotNodes, humanNodes);

  const robotByName = Object.fromEntries(robotNodes.map((n) => [n.name, n]));
  const humanByName = Object.fromEntries(humanNodes.map((n) => [n.name, n]));

  const traces = [];

  // Robot markers + labels
  traces.push({
    type: "scatter3d",
    mode: "markers+text",
    name: "robot links",
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
    type: "scatter3d", mode: "lines", name: "robot bones",
    legendgroup: "robot",
    x: rEdges.x, y: rEdges.y, z: rEdges.z,
    line: { color: C_BONE, width: 3 },
    hoverinfo: "skip",
  });

  const rAxes = axisSegments(robotNodes, [0, 0, 0]);
  for (const [key, color, label] of [
    ["x", "#e74c3c", "robot X"],
    ["y", "#27ae60", "robot Y"],
    ["z", "#3498db", "robot Z"],
  ]) {
    traces.push({
      type: "scatter3d", mode: "lines", name: label,
      legendgroup: "robot_axes", showlegend: key === "x",
      x: rAxes[key].x, y: rAxes[key].y, z: rAxes[key].z,
      line: { color, width: 3 }, hoverinfo: "skip",
    });
  }

  // Human markers + labels
  traces.push({
    type: "scatter3d",
    mode: "markers+text",
    name: "BVH bones",
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
    type: "scatter3d", mode: "lines", name: "human bones",
    legendgroup: "human",
    x: hEdges.x, y: hEdges.y, z: hEdges.z,
    line: { color: C_BONE, width: 3 },
    hoverinfo: "skip",
  });

  const hAxes = axisSegments(humanNodes, offset);
  for (const [key, color, label] of [
    ["x", "#e74c3c", "human X"],
    ["y", "#27ae60", "human Y"],
    ["z", "#3498db", "human Z"],
  ]) {
    traces.push({
      type: "scatter3d", mode: "lines", name: label,
      legendgroup: "human_axes", showlegend: key === "x",
      x: hAxes[key].x, y: hAxes[key].y, z: hAxes[key].z,
      line: { color, width: 3 }, hoverinfo: "skip",
    });
  }

  // Scene bounds
  const all = [];
  for (const n of robotNodes) all.push(n.pos);
  for (const n of humanNodes) {
    all.push([n.pos[0] + offset[0], n.pos[1] + offset[1], n.pos[2] + offset[2]]);
  }
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
      text: `${scene.xml_path.split("/").pop()}  ↔  ${scene.bvh_path.split("/").pop()}`,
      font: { size: 14 },
    },
    scene: {
      xaxis: { title: "X", range: [center[0] - extent, center[0] + extent], backgroundcolor: "#0b0f14", gridcolor: "#243040" },
      yaxis: { title: "Y", range: [center[1] - extent, center[1] + extent], backgroundcolor: "#0b0f14", gridcolor: "#243040" },
      zaxis: { title: "Z", range: [center[2] - extent, center[2] + extent], backgroundcolor: "#0b0f14", gridcolor: "#243040" },
      aspectmode: "cube",
      camera: { eye: { x: 1.25, y: 1.25, z: 0.85 } },
    },
    legend: {
      bgcolor: "rgba(20,26,34,0.85)",
      bordercolor: "#2c3642",
      font: { size: 11 },
      x: 0.01, y: 0.99,
    },
    uirevision: "gmr-stage1",
  };

  Plotly.react("plot", traces, layout, {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["toImage", "lasso2d", "select2d"],
  });
}

function payloadForSave() {
  return {
    mapping: mappingFromRows(),
    bvh_unit: Number($("bvhUnit").value),
    filename: $("saveName").value.trim() || "stage1_mapping.json",
    save_to_ik_configs: $("saveToIk").checked,
    base_config_path: state.baseConfigPath || null,
    robot_names: robotNames(),
    human_names: humanNames(),
    robot_root_name: $("robotRoot").value || null,
    human_root_name: $("humanRoot").value || null,
  };
}

async function previewConfig() {
  try {
    const data = await fetchJSON("/api/config/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadForSave()),
    });
    $("jsonPreview").textContent = data.json_text;
    $("jsonDialog").showModal();
  } catch (err) {
    $("saveStatus").textContent = String(err.message || err);
    $("saveStatus").classList.add("err");
  }
}

async function saveConfig() {
  $("saveStatus").classList.remove("err");
  try {
    const data = await fetchJSON("/api/config/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payloadForSave()),
    });
    $("saveStatus").textContent = `已保存: ${data.path}`;
    // refresh config list
    state.meta = await fetchJSON("/api/meta");
    const cur = $("configSelect").value;
    const cfgSel = $("configSelect");
    cfgSel.innerHTML = '<option value="">— 无 —</option>';
    for (const c of state.meta.ik_configs) {
      const opt = document.createElement("option");
      opt.value = c.path;
      opt.textContent = `${c.name}  (${c.dir})`;
      cfgSel.appendChild(opt);
    }
    cfgSel.value = cur;
    if (window.GmrSession) {
      GmrSession.capture();
      GmrSession.syncSaveName();
    }
  } catch (err) {
    $("saveStatus").textContent = String(err.message || err);
    $("saveStatus").classList.add("err");
  }
}

init().catch((err) => {
  $("saveStatus").textContent = "初始化失败: " + err;
  $("saveStatus").classList.add("err");
});
